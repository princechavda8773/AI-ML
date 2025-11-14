from google.colab import drive
drive.mount('/content/drive')

# ====================================================
# FINAL OPTIMIZED AUDIO CLASSIFICATION PIPELINE (99%+)
# ====================================================

# STEP 1: IMPORT LIBRARIES
# ====================================================
import os, random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from google.colab import files

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ====================================================
# STEP 2: SAFE DATA AUGMENTATION
# ====================================================
def augment_audio(audio, sr):
    """Apply random augmentation safely."""
    if len(audio) < sr * 0.1 or np.allclose(audio, 0):
        return audio  # skip short/silent clips

    choice = random.choice(["none", "time_stretch", "pitch_shift", "noise"])
    try:
        if choice == "time_stretch":
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate)
        elif choice == "pitch_shift":
            if len(audio) > sr * 0.25:
                n_steps = random.uniform(-1.5, 1.5)
                audio = librosa.effects.pitch_shift(audio, sr, n_steps)
        elif choice == "noise":
            noise = np.random.normal(0, 0.003, len(audio))
            audio = audio + noise
    except:
        pass
    return audio

# ====================================================
# STEP 3: FEATURE EXTRACTION
# ====================================================
def extract_features(file_path, n_mfcc=20, max_len=5.0, sr=22050, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        audio, _ = librosa.effects.trim(audio)

        if augment:
            audio = augment_audio(audio, sr)

        # Pad or crop
        max_samples = int(max_len * sr)
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Feature extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(mfccs_delta, axis=1), np.std(mfccs_delta, axis=1),
            np.mean(mfccs_delta2, axis=1), np.std(mfccs_delta2, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spec_centroid), np.std(spec_centroid),
            np.mean(spec_bw), np.std(spec_bw),
            np.mean(zcr), np.std(zcr)
        ])

        features = np.nan_to_num(features)
        return features
    except:
        return None

# ====================================================
# STEP 4: LOAD DATASETS
# ====================================================
def load_train_dataset(train_path, augment_factor=1):
    X, y, file_names = [], [], []
    for label_name in sorted(os.listdir(train_path)):
        folder = os.path.join(train_path, label_name)
        if not os.path.isdir(folder):
            continue
        count = 0
        for file in os.listdir(folder):
            if file.lower().endswith(".wav"):
                fpath = os.path.join(folder, file)
                feat = extract_features(fpath, augment=False)
                if feat is not None:
                    X.append(feat)
                    y.append(label_name)
                    file_names.append(file)
                    count += 1
                for _ in range(augment_factor):
                    feat_aug = extract_features(fpath, augment=True)
                    if feat_aug is not None:
                        X.append(feat_aug)
                        y.append(label_name)
                        file_names.append(file + "_aug")
        print(f"✅ Loaded {count} files from {label_name}")
    return np.array(X), np.array(y), np.array(file_names)

def load_test_dataset(test_path):
    X, file_names = [], []
    for file in sorted(os.listdir(test_path)):
        if file.lower().endswith(".wav"):
            fpath = os.path.join(test_path, file)
            feat = extract_features(fpath, augment=False)
            if feat is not None:
                X.append(feat)
                file_names.append(file)
    print(f"✅ Loaded {len(X)} test files")
    return np.array(X), file_names

# ====================================================
# STEP 5: LOAD TRAIN & TEST
# ====================================================
train_path = "/content/drive/MyDrive/the-frequency-quest/train/train"
test_path = "/content/drive/MyDrive/the-frequency-quest/test/test"

X_full, y_full, train_files = load_train_dataset(train_path, augment_factor=2)
X_test, test_files = load_test_dataset(test_path)

print(f"\nTraining data shape: {X_full.shape}")
print(f"Test data shape: {X_test.shape}")

# ====================================================
# STEP 6: NORMALIZATION + ENCODE LABELS
# ====================================================
scaler = StandardScaler()
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

encoder = LabelEncoder()
y_full = encoder.fit_transform(y_full)

X_train, X_val, y_train, y_val, _, _ = train_test_split(
    X_full, y_full, train_files, test_size=0.2, random_state=42, stratify=y_full
)

# ====================================================
# STEP 7: TORCH CONVERSION
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ====================================================
# STEP 8: MODEL DEFINITION
# ====================================================
input_size = X_train.shape[1]
num_classes = len(encoder.classes_)

model = nn.Sequential(
    nn.Linear(input_size, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.45),

    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.35),

    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.25),

    nn.Linear(128, num_classes)
).to(device)

# ====================================================
# STEP 9: TRAINING SETUP
# ====================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=120
)

epochs = 120
patience = 30

# ====================================================
# STEP 10: TRAINING LOOP
# ====================================================
best_val_acc, patience_counter = 0.0, 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Mixup
        if random.random() < 0.3:
            lam = np.random.beta(0.4, 0.4)
            index = torch.randperm(xb.size(0)).to(device)
            xb = lam * xb + (1 - lam) * xb[index, :]
            yb_a, yb_b = yb, yb[index]
            outputs = model(xb)
            loss = lam * criterion(outputs, yb_a) + (1 - lam) * criterion(outputs, yb_b)
        else:
            outputs = model(xb)
            loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds_all, val_true_all = [], []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_outputs = model(xb)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_preds_all.extend(val_preds.cpu().numpy())
            val_true_all.extend(yb.cpu().numpy())
        val_acc = accuracy_score(val_true_all, val_preds_all)

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model improved and saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# ====================================================
# STEP 11: INFERENCE
# ====================================================
print("\n🚀 Generating predictions on test set...")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

test_loader = DataLoader(X_test, batch_size=32, shuffle=False)
predictions = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        outputs = torch.softmax(model(xb), dim=1)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

pred_labels = encoder.inverse_transform(predictions)

submission = pd.DataFrame({
    "filename": test_files,
    "label": pred_labels
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")

files.download("submission.csv")