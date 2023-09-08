import os
from ctcdecode import CTCBeamDecoder
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import torchmetrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from conformer import Conformer

###########################################################################################

class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


############################################################################################


class LibriSpeechDataset(Dataset):
    def __init__(self, audio_files, waveform_length, context_length, future_length, negative_waveform_length):
        self.audio_files = audio_files
        self.waveform_length = waveform_length
        self.context_length = context_length
        self.future_length = future_length
        self.negative_waveform_length = negative_waveform_length

    def __len__(self):
        return len(self.audio_files)

    def load_waveform(self, audio_path, waveform_length):
        waveform, _ = torchaudio.load(audio_path)
        if waveform.size(1) > waveform_length:
            start_idx = random.randint(0, waveform.size(1) - waveform_length)
            waveform = waveform[:, start_idx: start_idx + waveform_length]
        else:
            pad_length = waveform_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.load_waveform(audio_path, self.waveform_length)

        # Generate context waves
        start_idx = random.randint(0, self.waveform_length - self.context_length - self.future_length)
        context = waveform[:, start_idx: start_idx + self.context_length]

        # Generate future samples
        future = waveform[:, start_idx + self.context_length: start_idx + self.context_length + self.future_length]

        # Generate negative sample
        negative_idx = random.randint(0, len(self.audio_files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.audio_files) - 1)

        negative_audio_path = self.audio_files[negative_idx]
        negative_waveform = self.load_waveform(negative_audio_path, self.negative_waveform_length)

        negative_sample = negative_waveform

        # Return context, future, negative sample, and waveform length
        return context, future, negative_sample, context.size(1)

###########################################################################################


import sentencepiece as spm

class SentencePieceTransform:
    """Maps subwords to integers and vice versa using SentencePiece"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_int(self, text):
        """ Use the SentencePiece tokenizer to convert text to an integer sequence """
        subwords = self.sp.EncodeAsPieces(text.lower())
        return [self.sp.PieceToId(subword) for subword in subwords]

    def int_to_text(self, labels):
        """ Use the SentencePiece tokenizer to convert integer labels to a text sequence """

        return self.sp.decode(labels)

sentencepiece_transform =  SentencePieceTransform("/home/exx/Desktop/conformer/spm_model1000.model")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=25),
#     RandomApply([PolarityInversion()], p=0.8),
#     RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
#     RandomApply([Gain()], p=0.2),                        
#     torchaudio.transforms.SlidingWindowCmn(cmn_window=500, center=True, norm_vars=False)

)

valid_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
#     torchaudio.transforms.SlidingWindowCmn(cmn_window=500, center=True, norm_vars=False)

)

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(sentencepiece_transform.text_to_int(utterance))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def numtoword(beam_results, out_lens, labels, label_lengths,blank_label=0, collapse_repeated=True):
    arg_maxes = beam_results

    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):
        decode = []
        tar_list = labels[i][:label_lengths[i]].tolist()
        tar_list = list(map(int, tar_list))
        tar_list = list(filter(lambda x: x != 0, tar_list))
        targets.append(sentencepiece_transform.int_to_text(tar_list))
    
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
        decodes.append(sentencepiece_transform.int_to_text(decode))
    return decodes, targets


###################################################################################


def loss_F(parameters):
    return sum(torch.linalg.norm(w) ** 2 for w in parameters)


    
    
loss_fn = InfoNCE()   
    
def train(model, premodel, device, train_loader, train_loader2, criterion, optimizer, preoptimizer,
           epoch, gam, optimizer1, preoptimizer1):
    model.train()
    premodel.train()
    
    train_loss = 0
    info_loss = 0
    
##     print("Model's state_dict:")
    
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    data_len = len(train_loader.dataset)
    data_len2 = len(train_loader2.dataset)
    
#     for batch_idx, (_data, inputs) in enumerate(zip(train_loader, train_loader2)):

    for batch_idx, (context, future, negative_samples, lengths) in enumerate(train_loader2):
        # Move batch tensors to device
            context = context.to(device)
            future = future.to(device)
            negative_samples = negative_samples.to(device)

#             print(context.size())
            # Forward pass
#             context = context.unsqueeze(1)#torch.squeeze(context, dim=1)
            
            context = context.repeat(1, 80, 1)
            
            context = context.transpose(1,2)
            
#             print(context.size())
                
            input_lengths=torch.LongTensor(lengths).to(device)
#             print(context.size())

#             future = future.unsqueeze(1)
#             future = future.repeat(1, 1, 80, 1)

#             negative_samples = negative_samples.unsqueeze(1)
#             negative_samples = negative_samples.repeat(1, 1, 80, 1)
        
        ########################unsupervised trainng portion####################################
            
#             print(inputs.size())
            


            predictions,_ = premodel(context, input_lengths)
            
#             predictions = torch.nn.functional.softmax(predictions,dim=2)
            
#             print(predictions.size())
            
#             target = premodel(future)
#             print(predictions.size())
#             print(target_segments.size())

#             neg_target = premodel(negative_samples)

            predictions = predictions[:, -1:, :]

            sizes = predictions.size()
        
#             print(predictions.size())
            
#             print(sizes)
#             print(future.size())
#             print(negative_samples.size())

            predictions = predictions.view(sizes[0], sizes[1]*sizes[2])

            target = future.view(sizes[0], sizes[1]*sizes[2])
            
            neg_target = negative_samples.view(sizes[0], sizes[1]*sizes[2])

            reg =  loss_F(premodel.parameters())  #torch.norm(predictions)**2
    
            loss_cpc = loss_fn(predictions, target, neg_target) + lamda*reg  # gxy
            

            # Backward and optimize
            
            preoptimizer.zero_grad()
            optimizer1.zero_grad()
            
            loss_cpc.backward()
            
            torch.nn.utils.clip_grad_norm_(parameters=premodel.parameters(), max_norm=1, norm_type=2.0)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2.0)
            
            preoptimizer.step()
            optimizer1.step()
            
            
            predi,_ = premodel(context, input_lengths)
#             targ = premodel(future)
#             neg_targ = premodel(negative_samples)
            predi = predi[:, -1:, :]
    
            predi = predi.view(sizes[0], sizes[1]*sizes[2])
#             targ =  targ.view(sizes[0], sizes[1]*sizes[2])
#             neg_targ =  neg_targ.view(sizes[0], sizes[1]*sizes[2])

#             reg1 = loss_F(premodel.parameters()) 
            
            
    
            loss_cpcctc = loss_fn(predi, target, neg_target) + lamda*reg
            info_loss += loss_cpcctc.item() / len(train_loader2)
        
            if batch_idx % 100 == 0 or batch_idx == data_len2:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCPC_Loss: {:.6f}'.format(
                    epoch, batch_idx * len(context), data_len2,
                    100. * batch_idx / len(train_loader2), loss_cpcctc.item()))
                
    print(f'Info_loss: {info_loss}')    
        ######################Supervised training portion#########################################
    for batch_idx, _data in enumerate(train_loader):
            preoptimizer1.zero_grad()
            
            optimizer.zero_grad()
#            
            
            gam = round(gam, 3)
        
            spectrograms, labels, input_lengths, label_lengths = _data 
            
            #print(input_lengths)
            
            spectrograms=torch.squeeze(spectrograms, dim=1)
            
#             print(spectrograms.size())
            
            spectrograms = spectrograms.transpose(1,2)
            
#             print(spectrograms.size())
            
            labels= torch.LongTensor(labels.long())
            
            input_lengths=torch.LongTensor(input_lengths)
            label_lengths=torch.LongTensor(label_lengths)
#             print(label_lengths.type())
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output, output_lengths = model(spectrograms,input_lengths)  # (batch, time, n_class)
            
            output = output.transpose(0, 1) # (time, batch, n_class)
            

            loss_ctc = criterion(output, labels, output_lengths, label_lengths) + gam*info_loss  #(fy + gam* (gxy-vx))
            
            lr_decay = min(1/(gam+1e-8),1)
            
            loss = lr_decay*loss_ctc   #(fy + gam* (gxy-vx))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=premodel.parameters(), max_norm=1, norm_type=2.0)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2.0)
            preoptimizer1.step()
            
            optimizer.step() 
            
    
            
            
            train_loss += loss_ctc.item() / len(train_loader)
            

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCTC_Loss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss_ctc.item()))
                
                print(f'gamma: {gam}')
    print(f'train_loss: {train_loss}') 
    
    
    
#     print(epoch)
    
#    if epoch==100:
#        
#        torch.save(model.state_dict(), '/home/ec2-user/SageMaker/conformer960model.pth')
    
    return train_loss, info_loss


def test(model, device, test_loader, criterion, epoch, batch_size=80):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    n_classes = 1000

    if epoch%20==0:
        with torch.no_grad():
                for i, _data in enumerate(test_loader):
                    spectrograms, labels, input_lengths, label_lengths = _data
                    
                    spectrograms=torch.squeeze(spectrograms, dim=1)
                    
                    spectrograms = spectrograms.transpose(1,2)
            
                    labels=labels.long()

                    input_lengths=torch.LongTensor(input_lengths)
                    label_lengths=torch.LongTensor(label_lengths)
                    input_lengths = input_lengths
                    label_lengths = label_lengths

                    spectrograms, labels = spectrograms.to(device), labels.to(device)

                    output, output_lengths = model(spectrograms,input_lengths)  # (batch, time, n_class)
                    soft_max = torch.nn.functional.softmax(output,dim=2)

                    output = output.transpose(0, 1) # (time, batch, n_class)
                    loss = criterion(output, labels, output_lengths, label_lengths)
                    test_loss += loss.item() / len(test_loader)
        
                    itera = spectrograms.size()
                    
            
                    decoder = CTCBeamDecoder(
                        [''] * (n_classes - 1) + [' '],
                        model_path=None,
                        alpha=0,
                        beta=0,
                        cutoff_top_n=40,
                        cutoff_prob=1.0,
                        beam_width=100,
                        num_processes=4,
                        blank_id=0,
                        log_probs_input=False
                    )
                    beam_results, beam_scores, timesteps, out_lens = decoder.decode(soft_max, output_lengths)
                    b=[]
                    for i in range(itera[0]):
                         b.append(beam_results[i][0][:out_lens[i][0]])
                    decoded_preds, decoded_targets = numtoword(b,out_lens,labels, label_lengths)
  
                    for j in range(len(decoded_preds)):
                        test_cer.append(torchmetrics.functional.char_error_rate(decoded_targets[j], decoded_preds[j]))
                        test_wer.append(torchaudio.functional.edit_distance(decoded_targets[j], decoded_preds[j]) / len(
    decoded_targets[j]
))                    

        avg_cer = sum(test_cer)/len(test_cer)
        avg_wer = sum(test_wer)/len(test_wer)

        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
        
        file_path = "/home/exx/Desktop/conformer/wer.txt"
        with open(file_path, "a") as file:
            file.write(f"Epoch {epoch}: {avg_wer}\n")

        return test_loss, avg_cer, avg_wer 
    #     return beam_results, out_lens, output
    else:
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                
                spectrograms=torch.squeeze(spectrograms, dim=1)
                
#                 print(spectrograms.size())
                
                spectrograms = spectrograms.transpose(1,2)
                
#                 print(spectrograms.size())
            
                labels=labels.long()

                input_lengths=torch.LongTensor(input_lengths)
                label_lengths=torch.LongTensor(label_lengths)
                
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)
                

                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output, output_lengths = model(spectrograms,input_lengths)  # (batch, time, n_class)
                soft_max = torch.nn.functional.softmax(output,dim=2)
#                     output_lengths = torch.full((output.size(0),), output.size(1), dtype=torch.int32)
#                     output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)
                loss = criterion(output, labels, output_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)
        print('Test set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss, 0 , 0
      

def main(learning_rate=5e-4, batch_size=80, epochs=10,
        train_url="train-clean-100", test_url="test-clean"):

    
    hparams = {
        "n_class": 1000,
        "n_feats": 80,
        "stride":2,
        "dropout": 0.05,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }



    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

   

    #splits = ["train-clean-100", "train-clean-360", "train-other-500"]

#    train_dataset1 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[0], download=True)
#    train_dataset22 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[1], download=True)
#    train_dataset3 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[2], download=True)
#    # Combine the dataset splits into a single dataset
#    combined_dataset = data.ConcatDataset([train_dataset1, train_dataset22, train_dataset3])
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,#combined_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size= hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    model = Conformer(num_classes=hparams['n_class'], 
                  input_dim=hparams['n_feats'], 
                  encoder_dim=512, 
                  num_encoder_layers=7)
    model.to(device)
    model = nn.DataParallel(model)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), lr=hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=0).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    optimizer1 = optim.AdamW(model.parameters(), lr=5e-3)
    

    
    ####################################Pre training######################################
    
    data_dir = "\LibriSpeech" # unsupervised training dataset directory

    audio_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            audio_files.append(os.path.join(root, file))
    
    waveform_length = 16000  # Length of the waveform (can be adjusted as needed)
    context_length = 256  # Length of the context wave
    future_length = 100  # Length of the future samples
    negative_waveform_length = 100

    train_dataset2 = LibriSpeechDataset(audio_files, waveform_length, context_length, future_length, negative_waveform_length)
      # Adjust the batch size as needed
    train_loader2 = DataLoader(train_dataset2, batch_size=hparams['batch_size']) # Iterate over the data loader
    
    print(len(train_loader.dataset))
    print(len(train_loader2.dataset))

    prehparams = {
        "n_class": 512,
        "n_feats": 80,
        "stride":2,
        "dropout": 0.05,
        "epochs": epochs
            }

    premodel = Conformer(num_classes=future_length, 
                  input_dim=80, 
                  encoder_dim=512, 
                  num_encoder_layers=7)

    
    preoptimizer = optim.AdamW(premodel.parameters(), lr=5e-3)
    prescheduler = optim.lr_scheduler.OneCycleLR(preoptimizer, max_lr=5e-3, 
                                            steps_per_epoch=int(len(train_loader2)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    
    preoptimizer1 = optim.AdamW(premodel.parameters(), lr=hparams['learning_rate'])

    premodel.to(device)
    premodel = nn.DataParallel(premodel)
    
    gamma_max = 1
    gamma_init = 0
    gamma_argmax_step = 500
    if gamma_init > gamma_max:
        gamma_max = gamma_init
        print('Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init.')
    gam = gamma_init
    step_gam = (gamma_max-gamma_init)/gamma_argmax_step
    
    train_loss=[]
    test_loss=[]
    Info_loss = []
    cer=[]
    wer=[]
    tes_loss1=3
    for epoch in range(1, epochs + 1):
        
        tra_loss, infoloss = train(model, premodel, device, train_loader, train_loader2, criterion, optimizer, preoptimizer, epoch, gam, optimizer1, preoptimizer1)
      
        prescheduler.step()
        scheduler.step()
        
        gam+= step_gam

        gam = min(gamma_max,gam)
        
        tes_loss, c, w =  test(model, device, test_loader, criterion, epoch)
        
        if tes_loss<tes_loss1:
            tes_loss1=tes_loss
            torch.save(model.state_dict(), '/home/exx/Desktop/saif/conformer/biconformer10model.pth')
        
#         scheduler.step(tes_loss)
        train_loss.append(tra_loss)
        test_loss.append(tes_loss)
        Info_loss.append(infoloss)
        cer.append(c)
        wer.append(w)
    return train_loss, test_loss, cer, wer, Info_loss

################################################################################################


learning_rate = 5e-4
batch_size = 80
epochs = 100
libri_train_set = "train-clean-100"
libri_test_set = "test-other"

train_loss, test_loss, cer, wer, Info_loss = main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set)
