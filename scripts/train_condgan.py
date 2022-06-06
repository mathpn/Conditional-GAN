"""
Train vanilla GAN with 102 Category Flower Dataset.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import utils

from utils.dataset import FlowerDataset
from models.generator import CondGenerator
from models.discriminator import CondDiscriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/102flowers')
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--input-dim', type=int, default=100, help='number of dimensions of generator input vector')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--embedding-dim', type=int, default=128, help='number of dimensions of embedding vector')
    parser.add_argument('--noise-factor', type=float, default=0.5, help='factor to scale added noise')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer beta1')
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()

    dataset = FlowerDataset(data_path=args.data_path, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator = CondGenerator(args.input_dim, 102, args.embedding_dim).to(device)
    discriminator = CondDiscriminator(args.img_size, 102, args.embedding_dim).to(device)

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_random_noise = torch.randn(64, args.input_dim, 1, 1, device=device)
    fixed_random_labels = torch.randperm(102).long()[:64].to(device)

    print('start training')
    criterion = nn.BCELoss()
    disc_loss_history = []
    gen_loss_history = []

    for epoch in range(args.epochs):
        start_time = time.time()
        disc_loss_obj, gen_loss_obj = 0.0, 0.0
        for i, (x, labels) in enumerate(loader):
            x = x.to(device)
            labels = labels.to(device)

            # train discriminator
            discriminator.zero_grad()
            real_out = discriminator(x + torch.rand_like(x) * 0.1 * args.noise_factor, labels).view(-1)
            real_loss = criterion(real_out, torch.ones_like(real_out))
            real_loss.backward()
            disc_loss_obj += real_loss.item()

            random_noise = torch.randn(x.shape[0], args.input_dim).to(device)
            fake = generator(random_noise, labels)
            fake_out = discriminator(fake.detach() + torch.rand_like(x) * 0.1 * args.noise_factor, labels).view(-1)
            fake_loss = criterion(fake_out, torch.zeros_like(fake_out))
            fake_loss.backward()
            disc_loss_obj += fake_loss.item()
            optimizer_discriminator.step()

            # train generator
            generator.zero_grad()
            fake_out = discriminator(fake + torch.rand_like(x) * 0.1 * args.noise_factor, labels).view(-1)
            gen_loss = criterion(fake_out, torch.ones_like(fake_out))
            gen_loss.backward()
            gen_loss_obj += gen_loss.item()
            optimizer_generator.step()

            if (i + 1) % 20 == 0:
                print(
                    f'Epoch: {epoch + 1}, Step: {i + 1} | Disc Loss: {disc_loss_obj/20:.4f} | Gen Loss: {gen_loss_obj/20:.4f}'
                )
                disc_loss_history.append(disc_loss_obj/20)
                gen_loss_history.append(gen_loss_obj/20)
                disc_loss_obj, gen_loss_obj = 0.0, 0.0
        print(f'epoch {epoch + 1} finished in {time.time() - start_time:.2f} seconds')

        # save sample images
        fixed_generated_imgs = generator(fixed_random_noise, fixed_random_labels).detach().cpu()
        fixed_generated_imgs = utils.make_grid(fixed_generated_imgs, nrow=8, normalize=True, padding=2)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(fixed_generated_imgs.cpu().detach().numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f'data/condgan_epoch_{epoch + 1}.png')
        plt.close()

        # save model
        if (epoch + 1) % 2 == 0:
            torch.save(generator.state_dict(), f'checkpoints/cond_generator_{epoch + 1}.pt')
            torch.save(discriminator.state_dict(), f'checkpoints/cond_discriminator_{epoch + 1}.pt')
    
    # save loss history plot
    plt.figure(figsize=(12, 4))
    plt.plot(disc_loss_history, label='discriminator loss')
    plt.plot(gen_loss_history, label='generator loss')
    plt.legend()
    plt.savefig('data/condgan_loss_history.png')
    plt.close()


if __name__ == '__main__':
    main()
