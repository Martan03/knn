from copy import deepcopy
from typing import OrderedDict

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from rich.progress import track
from torch.utils.data import DataLoader

from src.diffusion import create_diffusion
from src.loader import IAMDataset, collate_fn_padd
from src.models.dit import DiT_S_8
from src.models.encoders import LabelEncoder


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_size = 256
        latent_size = img_size // 8
        self.model = DiT_S_8(input_size=latent_size).to(self.device)

        self.ema = deepcopy(self.model).to(self.device)
        requires_grad(self.ema, False)
        self.ema.eval()

        self.diffusion = create_diffusion(timestep_respacing="")
        vae_type = "ema"
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}")
        self.vae = vae.to(self.device)

        self.label_enc = LabelEncoder(0.1).to(self.device)
        self.label_enc.initialize_weights()

        self.opt = torch.optim.AdamW(
            [
                {"params": self.model.parameters()},
                {"params": self.label_enc.parameters()},
            ],
            lr=1e-4,
            weight_decay=0,
        )

        label_path = args.dataset / "IAM64_train.txt"
        data_path = args.dataset / "IAM64-new/train"
        dataset = IAMDataset(label_path, data_path)
        self.loader = DataLoader(
            dataset,
            batch_size=args.batch,
            collate_fn=lambda x: collate_fn_padd(x, self.device),
        )

        self.epochs = args.epochs

    def train(self):
        update_ema(self.ema, self.model, decay=0)
        self.model.train()
        self.label_enc.train()

        for epoch in range(self.epochs):
            loss = self.train_epoch()
            print(f"epoch {epoch} loss {loss}")

        self.model.eval()

    def train_epoch(self) -> float:
        loss_sum = 0
        for d in track(self.loader, description="training"):
            x = d["expected"]
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(
                0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device
            )
            y = self.label_enc.text_transform(d["transcript"], self.device)
            model_kwargs = {
                "content": y,
                "style": d["style"],
            }

            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            update_ema(self.ema, self.model)

            loss_sum += loss.item()

        return loss_sum / len(self.loader) if len(self.loader) != 0 else float("inf")

    def save(self, file):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(checkpoint, file)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
