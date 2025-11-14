# !pip install torch torchaudio torchvision transformers

# ==============================================================================
# 0. IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torch.autograd import grad

# --- Import the HiFi-GAN Vocoder ---
# We use the prototype pipelines, which are very convenient
try:
    from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle
except ImportError:
    print("Could not import torchaudio.prototype.pipelines. Please ensure you have a recent version of torchaudio.")
    # You might need to run: !pip install --pre torchaudio -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html

# from google.colab import drive
# drive.mount('/content/drive')

# ==============================================================================
# 1. DATASET
# ==============================================================================
class TrainAudioSpectrogramDataset(Dataset):
    """
    MODIFIED: This dataset class now takes n_mels as an argument
    to match the HiFi-GAN vocoder's requirements (80 mels).
    """
    def __init__(self, root_dir, categories, n_mels=80, max_frames=512, fraction=1.0):
        self.root_dir = root_dir
        self.categories = categories
        self.max_frames = max_frames
        self.n_mels = n_mels
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}

        # Pre-calculate the Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=self.n_mels
        )

        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".wav")]
            num_to_sample = int(len(files_in_cat) * fraction)
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
       
        # Ensure correct sample rate (resample if needed, though HiFi-GAN bundle is 22050)
        if sr != 22050:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(wav)
            sr = 22050

        mel_spec = self.mel_transform(wav)
        log_spec = torch.log1p(mel_spec) # log1p(x) = log(1 + x)

        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return log_spec, label_vec

# ==============================================================================
# 2. WGAN-GP MODEL DEFINITIONS (GENERATOR & CRITIC)
# ==============================================================================
class CGAN_Generator(nn.Module):
    """
    MODIFIED: The Forger/Artist
    Architecture changed to output (80, 512) spectrograms.
    """
    def __init__(self, latent_dim, num_categories, spec_shape=(80, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        # Upsampling architecture to target 80x512
        # We start from 5x32 and upsample 4 times (x2) -> 80x512
        self.fc = nn.Linear(latent_dim + num_categories, 256 * 5 * 32)
        self.unflatten_shape = (256, 5, 32) # (channels, H, W)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 10x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 20x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 40x256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # -> 80x512
            nn.ReLU() # Use ReLU to match the log1p output range [0, inf)
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec

class CGAN_Discriminator(nn.Module):
    """
    MODIFIED: The Critic (not Detective)
    Architecture changed to accept (80, 512) spectrograms.
    BatchNorm layers are REMOVED (crucial for WGAN-GP).
    Final layer is linear (outputs a raw score, not a logit).
    """
    def __init__(self, num_categories, spec_shape=(80, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape

        # Embedding for the label to match the image dimensions
        self.label_embedding = nn.Linear(num_categories, H * W)

        # Downsampling architecture
        self.net = nn.Sequential(
            # Input channel is 2: 1 for spectrogram, 1 for label map
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1), # -> 40x256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 20x128
            # nn.BatchNorm2d(64), # REMOVED for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 10x64
            # nn.BatchNorm2d(128), # REMOVED for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 5x32
            # nn.BatchNorm2d(256), # REMOVED for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),

            # Final output layer to produce a single score (not a logit)
            nn.Conv2d(256, 1, kernel_size=(5, 32), stride=1, padding=0) # -> 1x1
        )

    def forward(self, spec, y):
        # Create a channel for the label and concatenate it with the spectrogram
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)

        # Pass through the network
        score = self.net(h)
        return score.view(-1, 1) # Flatten to a single value per item in batch

# ==============================================================================
# 3. WGAN-GP UTILITY FUNCTION (GRADIENT PENALTY)
# ==============================================================================

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device, lambda_gp):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    # Get random interpolation of images
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
   
    critic_interpolates = critic(interpolates, labels)
   
    # Use ones_like because we are measuring the gradient w.r.t. the interpolates
    fake_labels = torch.ones_like(critic_interpolates, device=device, requires_grad=False)
   
    # Get gradient w.r.t. interpolates
    gradients = grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
   
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gradient_penalty


# ==============================================================================
# 4. UTILITY FUNCTIONS (GENERATION, SAVING)
# ==============================================================================

def generate_audio_gan(generator, vocoder, category_idx, num_samples, device):
    """
    MODIFIED: Uses the HiFi-GAN vocoder instead of GriffinLim.
    """
    generator.eval()
    vocoder.eval()
   
    num_categories = generator.num_categories
    latent_dim = generator.latent_dim

    # Prepare label and noise
    y = F.one_hot(torch.tensor([category_idx]), num_classes=num_categories).float().to(device)
    y = y.repeat(num_samples, 1) # Repeat label for batch
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        log_spec_gen = generator(z, y) # [B, 1, 80, 512]

    # Convert log-spectrogram (log1p) back to linear-magnitude spectrogram
    # The vocoder expects a linear mel-spectrogram
    spec_gen = torch.expm1(log_spec_gen) # [B, 1, 80, 512]
    spec_gen = spec_gen.squeeze(1) # [B, 80, 512]

    # --- REMOVED GriffinLim and InverseMelScale ---

    # --- ADDED HiFi-GAN Vocoder ---
    with torch.no_grad():
        # The vocoder takes the batch of spectrograms and synthesizes audio
        waveform = vocoder(spec_gen) # [B, 1, T]

    return waveform.cpu()

def save_and_play(wav, sample_rate, filename):
    if wav.dim() > 2: wav = wav.squeeze(0)
    torchaudio.save(filename, wav, sample_rate=sample_rate)
    print(f"Saved to {filename}")
    display(Audio(data=wav.numpy(), rate=sample_rate))

# ==============================================================================
# 5. WGAN-GP TRAINING FUNCTION
# ==============================================================================
def train_gan(generator, discriminator, vocoder, dataloader, device, categories, epochs, lr, latent_dim, n_critic, lambda_gp):
   
    # WGAN-GP uses Adam with specific betas
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # REMOVED BCEWithLogitsLoss
   
    # Create directories for output
    os.makedirs("gan_generated_audio", exist_ok=True)
    os.makedirs("gan_spectrogram_plots", exist_ok=True)
   
    # Store losses
    G_losses = []
    D_losses = []

    print("--- Starting WGAN-GP Training ---")
   
    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
       
        for i, (real_specs, labels) in enumerate(loop):
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            batch_size = real_specs.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_D.zero_grad()
           
            # Generate fake data
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_specs = generator(z, labels)

            # Get critic scores
            real_output = discriminator(real_specs, labels)
            fake_output = discriminator(fake_specs.detach(), labels)

            # Compute gradient penalty
            gp = compute_gradient_penalty(discriminator, real_specs, fake_specs.detach(), labels, device, lambda_gp)
           
            # Compute WGAN-GP loss
            loss_D = torch.mean(fake_output) - torch.mean(real_output) + gp
           
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            # Only update Generator every n_critic iterations
            if i % n_critic == 0:
                optimizer_G.zero_grad()
               
                # Generate new fake specs
                fake_specs_gen = generator(z, labels) # No need for new z, reuse labels
               
                # Get critic score
                output_gen = discriminator(fake_specs_gen, labels)
               
                # Compute Generator loss
                loss_G = -torch.mean(output_gen)
               
                loss_G.backward()
                optimizer_G.step()
               
                # Save losses for plotting
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
            else:
                loop.set_postfix(loss_D=loss_D.item())


        # --- End of Epoch: Generate and save samples ---
        if epoch % 1 == 0:
            print(f"\n--- Generating Samples for Epoch {epoch} ---")
            generator.eval() # Set generator to eval mode for sampling

            # --- PLOTTING CODE ---
            fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
            if len(categories) == 1: axes = [axes] # Make it iterable

            for cat_idx, cat_name in enumerate(categories):
                y_cond = F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
                z_sample = torch.randn(1, generator.latent_dim).to(device)
                with torch.no_grad():
                    spec_gen_log = generator(z_sample, y_cond)

                spec_gen_log_np = spec_gen_log.squeeze().cpu().numpy()
                axes[cat_idx].imshow(spec_gen_log_np, aspect='auto', origin='lower', cmap='viridis')
                axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
                axes[cat_idx].axis('off')

            plt.tight_layout()
            plt.savefig(f'gan_spectrogram_plots/epoch_{epoch:03d}.png')
            plt.show()
            plt.close(fig) # Close the figure to free up memory

            # --- Audio generation (using HiFi-GAN) ---
            for cat_idx, cat_name in enumerate(categories):
                wav = generate_audio_gan(generator, vocoder, cat_idx, 1, device)
                fname = f"gan_generated_audio/{cat_name}_ep{epoch}.wav"
                save_and_play(wav, sample_rate=22050, filename=fname)

            generator.train() # Set back to training mode
            print("--- End of Sample Generation ---\n")
           
    # --- End of Training: Plot Losses ---
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Losses (WGAN-GP)")
    plt.plot(G_losses, label="Generator")
    plt.plot([D_losses[i] for i in range(0, len(D_losses), n_critic)], label="Critic") # Plot D loss at same freq as G
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gan_loss_plot.png')
    plt.show()

# ===============================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LATENT_DIM = 100
    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 2e-4
   
    # --- NEW WGAN-GP & HiFi-GAN Config ---
    M_MELS = 80          # HiFi-GAN standard
    SPEC_FRAMES = 512
    SPEC_SHAPE = (M_MELS, SPEC_FRAMES)
    SAMPLE_RATE = 22050  # HiFi-GAN standard
    N_CRITIC = 5         # WGAN-GP: Train critic 5 times per generator update
    LAMBDA_GP = 10       # WGAN-GP: Gradient penalty coefficient

    # --- Paths and Data Setup ---
    BASE_PATH = '/kaggle/input/the-frequency-quest/the-frequency-quest - Copy/train'
    TRAIN_PATH = os.path.join(BASE_PATH, 'train')
    train_categories = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    NUM_CATEGORIES = len(train_categories)

    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CATEGORIES} categories: {train_categories}")
    print(f"Using HiFi-GAN Vocoder (requires n_mels={M_MELS})")

    # --- Load HiFi-GAN Vocoder ---
    print("Loading HiFi-GAN Vocoder...")
    vocoder = bundle.get_vocoder().to(DEVICE)
    vocoder.eval()
    print("Vocoder loaded.")

    # --- Initialize Dataset & Dataloader ---
    train_dataset = TrainAudioSpectrogramDataset(
        root_dir=TRAIN_PATH,
        categories=train_categories,
        n_mels=M_MELS, # Pass new n_mels
        max_frames=SPEC_FRAMES
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # --- Initialize Models ---
    generator = CGAN_Generator(LATENT_DIM, NUM_CATEGORIES, spec_shape=SPEC_SHAPE).to(DEVICE)
    discriminator = CGAN_Discriminator(NUM_CATEGORIES, spec_shape=SPEC_SHAPE).to(DEVICE)

    print("--- Model Architectures ---")
    print(f"Generator output shape: (1, {M_MELS}, {SPEC_FRAMES})")
    print(f"Critic input shape: (1, {M_MELS}, {SPEC_FRAMES})")
    print("-----------------------------")

    # --- Start Training ---
    train_gan(
        generator=generator,
        discriminator=discriminator,
        vocoder=vocoder,
        dataloader=train_loader,
        device=DEVICE,
        categories=train_categories,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP
    )