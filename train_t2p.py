"""
Train encoder-decoder architecture to get text-to-palette embeddings.
"""

import datetime
import wandb
import torch
import os 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_loader import Text2PaletteDataset
from palette_model import PaletteGeneration as t2p

CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 256,
    "num_epochs": 5000,
    "max_len": 10,
    "embedding_dim": 512,
    "num_blocks": 4,
    "reduce_factor": 2,
    "save_every": 250,
    "train_embeddings": "data/clip_embs/train_palette_prompts.pt",
    "train_trg_path": "data/hexcolor_vf/train_palettes_rgb.pkl",
    "test_embeddings": "data/clip_embs/test_palette_prompts.pt",
    "test_trg_path": "data/hexcolor_vf/train_palettes_rgb.pkl",
    "resume": False,
    "save_dir": "models/PaletteEncoder"
}

def get_latest_checkpoint(path):
    checkpoints = [f for f in os.listdir(path) if f.endswith('.pth')]
    if checkpoints:
        return max(checkpoints, key=lambda s: int(s.split(".")[0].split("_")[-1]))
    return None


# Initialize Weights and Biases
wandb.init(project="text-to-palette",
           entity="jh216",
           config=CONFIG)

# Load input dictionary
input_dict = wandb.config

# Get train and test data loaders
train_data = Text2PaletteDataset(input_dict.train_embeddings, input_dict.train_trg_path,
                                 input_dict.batch_size)\
                    .to_data_loader()

test_data = Text2PaletteDataset(input_dict.test_embeddings, input_dict.test_trg_path,
                                input_dict.batch_size)\
                    .to_data_loader()

# Initialize model
model = t2p.PaletteDecoder(n_blocks=input_dict.num_blocks,
                           in_channels=input_dict.embedding_dim, 
                           num_colors=5,
                           reduce_factor=input_dict.reduce_factor)

# Load checkpoint if resuming
if input_dict.resume:
    checkpoint_path = get_latest_checkpoint(input_dict.save_dir)
    if checkpoint_path:
        model.load_state_dict(torch.load(os.path.join(input_dict.save_dir, checkpoint_path)))
        print(f"Loaded checkpoint: {checkpoint_path}")

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n\n")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=input_dict.learning_rate)

# savefir for new checkpoints
today = datetime.datetime.now()
savedir = os.path.join(input_dict.save_dir, today.strftime("%Y%m%d-%H%M%S"))
os.makedirs(savedir, exist_ok=True)

# Training loop
for _e in range(input_dict.num_epochs):
    for i, (text, palette) in enumerate(train_data):
        # Move data to GPU
        text = text.to(device)
        palette = palette.to(device)

        # Forward pass
        outputs = model(text)

        # Compute loss
        loss = criterion(outputs, palette)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        if (i + 1) % 100 == 0:
            wandb.log({"epoch": _e + 1, "loss": loss.item()})
        
    print(f"Epoch [{_e + 1}/{input_dict.num_epochs}], Loss: {loss.item():.4f}")

    # evaluate every 250 steps
    if (_e + 1) % input_dict.save_every == 0:
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(savedir, f"model_epoch_{_e + 1}.pth"))

        # run on test data
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for text, palette in test_data:
                text = text.to(device)
                palette = palette.to(device)
                outputs = model(text)
                test_loss += criterion(outputs, palette).item()

            test_loss /= len(test_data)
            wandb.log({"test_loss": test_loss})