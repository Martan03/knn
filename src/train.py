import os
from copy import deepcopy
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from rich.progress import track
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.diffusion import create_diffusion
from src.loader import IAMDataset, collate_fn_padd
from src.models.dit import DiT_S_8
from src.models.encoders import LabelEncoder


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_dir = Path(args.result_dir)
        os.mkdir(self.result_dir)

        img_size = 256
        self.latent_size = img_size // 8
        self.model = DiT_S_8(input_size=self.latent_size).to(self.device)

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

        test_label_path = args.dataset / "IAM64_test.txt"
        test_data_path = args.dataset / "IAM64-new/test"
        self.test_dataset = IAMDataset(test_label_path, test_data_path)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch,
            collate_fn=lambda x: collate_fn_padd(x, self.device),
        )

        self.epochs = args.epochs

    def train(self):
        update_ema(self.ema, self.model, decay=0)
        self.model.train()
        self.label_enc.train()

        best_score = float("inf")
        best_loss = float("inf")

        for epoch in range(self.epochs):
            loss = self.train_epoch()
            self.save(self.result_dir / "last.pt")
            score = self.eval()
            if score <= best_score:
                best_score = score
                self.save(self.result_dir / "best.pt")
            if loss < best_loss:
                best_loss = loss
            print(f"epoch {epoch}")
            print(f"loss: {loss} best: {best_loss}")
            print(f"score: {score} best: {best_score}")

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

    @torch.no_grad()
    def eval(self) -> float:
        self.label_enc.eval()

        idx = np.random.randint(0, len(self.test_dataset))
        data = self.test_dataset[idx]
        self.sample(
            data["transcription"], data["style"], str(self.result_dir / "latest.png")
        )

        self.label_enc.train()

        return 0

        loss_sum = 0
        for d in track(self.test_loader, description="evaluating"):
            x = d["expected"]
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)

            y = self.label_enc.text_transform(d["transcript"], self.device)
            z = torch.randn(
                len(d["transcript"]),
                4,
                self.latent_size,
                self.latent_size,
                device=self.device,
            )
            model_kwargs = {
                "content": y,
                "style": d["style"],
            }

            samples = self.diffusion.p_sample_loop(
                self.model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=self.device,
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample

        return 0

    def sample(self, text: str, style: torch.Tensor, file: str):
        txt = self.label_enc.text_transform([text], self.device)
        z = torch.randn(
            1,
            4,
            self.latent_size,
            self.latent_size,
            device=self.device,
        )
        model_kwargs = {
            "content": txt,
            "style": style.unsqueeze(1),
        }

        samples = self.diffusion.p_sample_loop(
            self.ema.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )
        samples, _ = samples.chunk(2, dim=0)
        samples = self.vae.decode(samples / 0.18215).sample
        save_image(samples, file, nrow=4, normalize=True, value_range=(-1, 1))

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
