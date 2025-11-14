# Conditional WGAN-GP + HiFi-GAN Audio Generator

A PyTorch implementation of a class-conditioned WGAN-GP that generates Mel-spectrograms and converts them into high-quality audio using the HiFi-GAN vocoder. Designed for The Frequency Quest dataset.

---

## Features

- WGAN-GP training (stable, no mode collapse)
- Conditional generation using one-hot labels
- Mel-spectrogram size: 80 × 512 (HiFi-GAN compatible)
- HiFi-GAN vocoder for high-quality waveform synthesis
- Spectrogram and audio samples saved every epoch
- Loss graph saved after training

---

## Project Structure

```
📁 project
│── train.py
│── README.md
│── gan_generated_audio/
│── gan_spectrogram_plots/
│── gan_loss_plot.png
│
└── /data
    └── /train
         ├── class1
         ├── class2
         ├── ...
```

---

## Installation

```
pip install torch torchvision torchaudio transformers
```

If torchaudio version is old:

```
pip install --pre torchaudio
```

---

## Dataset Format

```
train/
  ├── dog/
  │     ├── 001.wav
  │     ├── 002.wav
  ├── rain/
  │     ├── 001.wav
  ├── fire/
```

Each folder name is treated as a category label.

---

## Run Training

```
python train.py
```

Automatically generated during training:

- `gan_spectrogram_plots/epoch_XXX.png`
- `gan_generated_audio/classname_epX.wav`
- `gan_loss_plot.png`

---

## Audio Generation (Example)

```python
wav = generate_audio_gan(generator, vocoder, category_idx=0, num_samples=1, device='cuda')
torchaudio.save("sample.wav", wav, 22050)
```

---

## Architecture Overview

### Generator
- Input: noise vector + one-hot label  
- Output: (1 × 80 × 512) Mel-spectrogram  
- Fully connected + 4× ConvTranspose layers

### Critic
- Input: spectrogram + label map channel  
- Output: scalar critic score  
- No sigmoid, no batch-norm  
- Gradient penalty applied

### Mel-spectrogram settings
- n_mels = 80  
- n_fft = 1024  
- hop_length = 256  
- sample_rate = 22050  

---

## Requirements

| Library | Version |
|---------|---------|
| PyTorch | 2.x |
| Torchaudio | 2.x |
| Matplotlib | latest |
| TQDM | latest |

---

## Future Improvements

- Diffusion-based Mel generator  
- Attention-based generator  
- Multi-style or multi-speaker conditioning  

---

## Contributing

Pull requests and issues are welcome.

---

## Star the Repository

If you find this project useful, please star it on GitHub!
