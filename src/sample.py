import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torchvision.utils import save_image

from src.diffusion import create_diffusion
from src.loader import prep_img
from src.models.dit import DiT_S_8


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    latent_size = 256 // 8

    ema = DiT_S_8(input_size=latent_size).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    ema.load_state_dict(checkpoint["ema"])

    diffusion = create_diffusion("250")

    style = prep_img(args.style)
    text = args.text

    txt = ema.y_embedder.text_transform([text], device)
    z = torch.randn(
        1,
        4,
        latent_size,
        latent_size,
        device=device,
    )
    model_kwargs = {
        "content": txt,
        "style": style.unsqueeze(1),
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

    pass
