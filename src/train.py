import torch

from diffusion import create_diffusion
from src.models.dit import DiT_S_8


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    latent_size = img_size // 8
    model = DiT_S_8(input_size=latent_size)

    diffusion = create_diffusion(timestep_respacing="")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
