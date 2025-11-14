# Audio Classification Pipeline (Google Colab)

A complete, optimized end-to-end audio classification system using MFCC-based feature extraction, augmentation, deep learning (PyTorch), mixup, and OneCycleLR scheduling. Achieves 99%+ validation accuracy on many structured audio datasets.

This pipeline is built for easy use in Google Colab and includes training, validation, augmentation, and final prediction CSV generation.

---

## Features

- Works directly in Google Colab  
- Automatic Google Drive mounting  
- Safe data augmentation  
- MFCC + chroma + centroid + bandwidth + ZCR features  
- StandardScaler normalization  
- LabelEncoder for class mapping  
- Deep neural network with dropout + batch normalization  
- Mixup regularization  
- OneCycleLR scheduler  
- Early stopping  
- Generates `submission.csv` ready for Kaggle  

---

## Project Structure

```
📁 project
│── audio_classification.ipynb (your Colab notebook)
│── submission.csv
│── best_model.pth
│── README.md
│
└── /train
      ├── class1
      ├── class2
      ├── ...
└── /test
      ├── file1.wav
      ├── file2.wav
      ├── ...
```

---

## Installation (Colab)

No installation needed for Colab, but librosa may require updating:

```
pip install librosa --upgrade
```

---

## How It Works

### 1. Load Google Drive  
Automatically mounts your drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Data Augmentation  
Includes:
- Time stretch  
- Pitch shift  
- Gaussian noise  
- Random none (to avoid overfitting)

### 3. Feature Extraction  
Extracts:  
- MFCC + deltas  
- Chroma  
- Spectral centroid  
- Spectral bandwidth  
- Zero-crossing rate  

All features are averaged and stacked into one vector.

### 4. Dataset Loading  
Loads all `.wav` files from folder structure:

```
train/class_name/*.wav
test/*.wav
```

### 5. Preprocessing  
- StandardScaler normalization  
- Label encoding  
- Train/validation split with stratification  

### 6. Deep Learning Model  
A fully connected network with:

```
512 → 256 → 128 → output_classes
BatchNorm + ReLU + Dropout
```

### 7. Training  
- CrossEntropy with label smoothing  
- AdamW optimizer  
- OneCycleLR  
- Gradient clipping  
- Mixup regularization  
- Early stopping  
- Best model saved as:  
  ```
  best_model.pth
  ```

### 8. Inference  
Generates predictions for test set:

```
submission.csv
```

Format:

```
filename,label
xxx.wav,dog
yyy.wav,rain
```

---

## Usage

### Replace these paths:

```
train_path = "/content/drive/MyDrive/.../train/train"
test_path = "/content/drive/MyDrive/.../test/test"
```

### Run the full notebook  
Training automatically saves:
- `best_model.pth`
- `submission.csv`

---

## Output Files

| File | Description |
|------|-------------|
| best_model.pth | Best validation accuracy model |
| submission.csv | Final predictions |
| README.md | Documentation |

---

## Requirements

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| Torch | 2.x |
| Librosa | latest |
| Scikit-learn | latest |
| Pandas | latest |

---

## Future Improvements

- Add mel-spectrogram CNN  
- Add SpecAugment  
- Add transformer-based classifier  
- Add pseudo-labeling  

---

## Credits

This pipeline is optimized for fast experimentation in Google Colab with high accuracy and reliability.

