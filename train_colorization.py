import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm
from data_loader import p2c_loader  # Assumes p2c_loader function loads data specifically for palette-to-colorization
import palette_model.ColorizationModel as PCN
from util import process_image, process_palette_lab, process_global_lab
import time 

def train_PCN(args):
    device = args.device
    
    start_time = time.time()
    # Load dataset
    print("Loading dataset...")
    train_loader, imsize = p2c_loader(args.batch_size, cap=args.cap_data_size)
    print("Dataset loaded. Took {} seconds.".format(time.time() - start_time))

    # Initialize GAN models
    G = PCN.UNet(imsize, args.add_L).to(device)
    D = PCN.Discriminator(args.add_L, imsize).to(device)

    # Define loss functions
    criterion_GAN = nn.BCELoss()
    criterion_smoothL1 = nn.SmoothL1Loss()

    # Optimizers and schedulers
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    g_scheduler = lr_scheduler.ReduceLROnPlateau(g_optimizer, 'min', patience=5, factor=0.1)
    d_scheduler = lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min', patience=5, factor=0.1)

    # Training loop
    print('Starting training...')
    start_time = time.time()
    for epoch in tqdm(range(args.num_epochs)):
        for _, (images, palettes) in enumerate(train_loader):
            # Process input data
            palettes = palettes.view(-1, 5, 3).cpu().data.numpy()
            inputs, real_images, global_hint = prepare_data(images, palettes, args.always_give_global_hint, device)
            batch_size = inputs.size(0)

            # Labels for GAN loss
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            # Train Discriminator
            D.zero_grad()
            _in_tensor = torch.cat((real_images, global_hint), dim=1)
            real_outputs = D(_in_tensor)

            # NOTE update real labels shape
            real_labels = real_labels.view(-1, 1)
            d_loss_real = criterion_GAN(real_outputs, real_labels)

            fake_images = G(inputs, global_hint)
            fake_outputs = D(torch.cat((fake_images, global_hint), dim=1))
            fake_labels = fake_labels.view(-1, 1)
            d_loss_fake = criterion_GAN(fake_outputs, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) * args.lambda_GAN
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            G.zero_grad()
            fake_images = G(inputs, global_hint)
            fake_outputs = D(torch.cat((fake_images, global_hint), dim=1))
            g_loss_GAN = criterion_GAN(fake_outputs, real_labels)

            outputs = fake_images.view(batch_size, -1)
            labels = real_images.contiguous().view(batch_size, -1)
            g_loss_smoothL1 = criterion_smoothL1(outputs, labels)

            g_loss = g_loss_GAN * args.lambda_GAN + g_loss_smoothL1
            g_loss.backward()
            g_optimizer.step()

            # Step the schedulers
            g_scheduler.step(g_loss)
            d_scheduler.step(d_loss)

        # Logging
        if (epoch + 1) % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f'Elapsed time [{elapsed_time:.4f}], Epoch [{epoch + 1}/{args.num_epochs}], '
                  f'd_loss: {d_loss.item():.6f}, g_loss: {g_loss.item():.6f}')

        # Save model checkpoints
        if (epoch + 1) % args.save_interval == 0:
            torch.save(G.state_dict(), os.path.join(args.pal2color_dir, f'{epoch + 1}_G.ckpt'))
            torch.save(D.state_dict(), os.path.join(args.pal2color_dir, f'{epoch + 1}_D.ckpt'))
            print('Model checkpoints saved.')

def prepare_data(images, palettes, always_give_global_hint, device):
    # NOTE: important: i'm keeping the add_L as True always here

    # Assumes process_image, process_palette_lab, and process_global_lab from util.py are available
    batch = images.size(0)
    imsize = images.size(3)

    inputs, labels = process_image(images, batch, imsize) # labels size [2,2,256,256] , inputs size [2,1,256,256]
    for_global = process_palette_lab(palettes, batch) # for_global size [2,15,1,1]
    global_hint = process_global_lab(for_global, batch, always_give_global_hint)

    inputs = inputs.to(device)
    labels = labels.to(device)
    global_hint = global_hint.expand(-1, -1, imsize, imsize).to(device)
    return inputs, labels, global_hint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str,  help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--lambda_GAN', type=float, default=10.0, help='Weight for GAN loss') # in the paper they use 10
    parser.add_argument('--add_L', action='store_true', default=True, help='Add L channel to input')
    parser.add_argument('--always_give_global_hint', action='store_true', default=True, help='Always provide global hint')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving models')
    parser.add_argument('--pal2color_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--cap_data_size', type=int, default=None, help='Cap the number of data samples')
    args = parser.parse_args()

    os.makedirs(args.pal2color_dir, exist_ok=True)
    train_PCN(args)