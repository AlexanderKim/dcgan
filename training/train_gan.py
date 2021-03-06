import os
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from tqdm.auto import tqdm

from datasets.celeba_local_dataset import load_dataset
from networks.Discriminator import Discriminator
from networks.Generator import Generator


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def _train_loop(data_loader: torch.utils.data.DataLoader, gen: Generator, disc: Discriminator,
                criterion: BCEWithLogitsLoss, gen_opt: torch.optim.Optimizer, disc_opt: torch.optim.Optimizer, n_epochs=5,
                feedback_fn: Callable[[torch.Tensor, torch.Tensor, float, float], None] = None):

    gen_losses = []
    disc_losses = []
    for epoch in range(n_epochs):
        cur_step = 0

        for real, _ in tqdm(data_loader):
            real = real.to('cuda')

            disc_opt.zero_grad()
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            fake = gen(gen.gen_noize())
            #         fake = gen(torch.randn(real.size(0), nz, 1, 1, device=device))

            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            disc_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()

            gen_fake_pred = disc(gen(gen.gen_noize()))
            #         gen_fake_pred = disc(gen(torch.randn(real.size(0), nz, 1, 1, device=device)))

            gen_fake_loss = criterion(gen_fake_pred, torch.ones_like(gen_fake_pred))

            gen_fake_loss.backward()
            gen_opt.step()

            gen_losses += [gen_fake_loss]
            disc_losses += [disc_loss]

            cur_step += 1
            if cur_step % 500 == 0 and feedback_fn:
                feedback_fn(fake, real, gen_losses, disc_losses) # TODO: async fire and forget?

    return gen, disc


def train_gan(data_path, save_gen_path, save_disc_path, feedback_fn=None):
    data_loader = load_dataset(path=data_path, batch_size=128)

    gen = Generator().to('cuda')
    disc = Discriminator().to('cuda')

    criterion = torch.nn.BCEWithLogitsLoss()
    lr = 3e-4
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    n_epochs = 5

    gen, disc = _train_loop(data_loader=data_loader,
                            gen=gen, disc=disc,
                            criterion=criterion,
                            gen_opt=gen_opt,
                            disc_opt=disc_opt,
                            n_epochs=5,
                            feedback_fn=feedback_fn)

    torch.save(gen.state_dict(), save_gen_path)
    torch.save(disc.state_dict(), save_disc_path)


if __name__ == "__main__":
    train_gan("../data", "gen", "disc")
