from pathlib import Path
from typing import Tuple

import Levenshtein
import numpy as np
import pytesseract
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from rich.progress import track
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from src.diffusion import create_diffusion
from src.loader import IAMDataset, collate_fn_padd, decode_img, prep_img
from src.models.dit import DiT_S_8
from src.models.style import StyleNet


class Sampler:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_size = 256 // 8

        checkpoint = torch.load(args.model, map_location=self.device)
        self.ema = DiT_S_8(input_size=self.latent_size).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.eval()

        self.diffusion = create_diffusion("250")

        style_checkpoint = torch.load(args.style_model, map_location=self.device)
        self.style_model = StyleNet(8).to(self.device)
        self.style_model.load_state_dict(style_checkpoint["model"])

        test_label_path = args.dataset / "IAM64_test.txt"
        test_data_path = args.dataset / "IAM64-new/test"
        self.test_dataset = IAMDataset(test_label_path, test_data_path)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch,
            collate_fn=lambda x: collate_fn_padd(x, self.device),
        )
        self.fid = FrechetInceptionDistance().to(self.device)

    def sample(self, img: Path, text: str) -> torch.Tensor:
        style = prep_img(img).to(self.device)

        txt = self.ema.y_embedder.text_transform([text], self.device)
        z = torch.randn(
            1,
            4,
            self.latent_size,
            self.latent_size,
            device=self.device,
        )
        z = torch.cat([z, z], 0)
        txt = {
            k: torch.cat([v, torch.zeros_like(v)], 0).to(self.device)
            for k, v in txt.items()
        }
        # txt = torch.cat([txt, torch.zeros_like(txt)], 0)
        style = style.unsqueeze(0)
        style = torch.cat([style, torch.ones_like(style)], 0)
        model_kwargs = {
            "content": txt,
            "style": style,
            "cfg_scale": 4.0,
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
        return samples

    def eval(self) -> Tuple[float, float, float]:
        total_diff = 0
        total_cer = 0

        self.fid.reset()
        for d in track(self.test_loader, description="evaluating"):
            gen = []
            for style in d["style"]:
                txt = self.test_dataset.rand_text()
                res_img = self.sample(style, txt)

                gen_res = torch.clamp((res_img + 1.0) / 2.0, 0.0, 1.0)
                gen_res = gen_res.permute(1, 2, 0)
                gen.append(gen_res)

                res_tensor = decode_img(res_img)
                res_style = self.style_model.forward(res_tensor.unsqueeze(0))
                ref_img = style.to(self.device)
                ref_style = self.style_model.forward(ref_img.unsqueeze(0))

                total_diff += torch.sum(
                    torch.abs(torch.sub(res_style, ref_style))
                ).item()

                result = tensor_to_img(res_tensor)
                res = pytesseract.image_to_string(result, config="--psm 7").strip()
                cer = get_cer([res], [txt])
                total_cer += cer

            self.fid.update(d["expected"], real=True)
            self.fid.update(torch.stack(gen), real=False)

        cnt = len(self.test_dataset)
        return total_diff / cnt, total_cer / cnt, self.fid.compute().item()


def tensor_to_img(src: torch.Tensor) -> Image.Image:
    img = src.detach().cpu()
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def get_cer(preds: list[str], targets: list[str]) -> float:
    total_dist = 0
    total_len = 0

    for pred, target in zip(preds, targets):
        dist = Levenshtein.distance(pred, target)
        total_dist += dist
        total_len += len(target)

    return total_dist / total_len if total_len > 0 else 0.0


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    latent_size = 256 // 8

    ema = DiT_S_8(input_size=latent_size).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    ema.load_state_dict(checkpoint["ema"])
    ema.eval()

    diffusion = create_diffusion("250")

    style = prep_img(args.style).to(device)
    text = args.text

    txt = ema.y_embedder.text_transform([text], device)
    z = torch.randn(
        1,
        4,
        latent_size,
        latent_size,
        device=device,
    )
    z = torch.cat([z, z], 0)
    txt = {k: torch.cat([v, torch.zeros_like(v)], 0).to(device) for k, v in txt.items()}
    # txt = torch.cat([txt, torch.zeros_like(txt)], 0)
    style = style.unsqueeze(0)
    style = torch.cat([style, torch.ones_like(style)], 0)
    model_kwargs = {
        "content": txt,
        "style": style,
        "cfg_scale": 4.0,
    }

    samples = diffusion.p_sample_loop(
        ema.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
