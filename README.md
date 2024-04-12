# JUST: Joint self-supervised and supervised training for speech models
We present a novel bilevel optimization training approach for ASR models that we term joint unsupervised and supervised training (JUST). JUST employs a lower-level optimization with an unsupervised loss and an upper-level optimization with a supervised loss, leveraging recent advances in penalty-based bilevel optimization.

<p align="center">
  <img src="Screenshot 2023-09-07 at 21-23-28 AIRC_asr.png" width="500" title="hover text">
</p>

# Dataset
1. LibriSpeech: https://www.openslr.org/12
2. Ted-Lium: https://www.openslr.org/19/

# Comformer Model
We have used conformer model from: https://github.com/sooftware/conformer

# InfoNCE Loss

We have used InfoNCE loss from: https://github.com/RElbers/info-nce-pytorch

# Acknowledgement
Cisco Research and IBM Research
