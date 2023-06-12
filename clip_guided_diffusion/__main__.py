"""Generates images from text prompts with CLIP guided diffusion."""

import argparse
from concurrent import futures
import hashlib
import math
from pathlib import Path

import clip
from guided_diffusion import script_util
import k_diffusion as K
import requests
from rich import print
from rich.align import Align
from rich.panel import Panel
import safetensors.torch
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm

print = tqdm.external_write_mode()(print)


def download_file(url, root, expected_sha256):
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(url).name

    if target.exists() and not target.is_file():
        raise RuntimeError(f"{target} exists and is not a regular file")

    if target.is_file():
        if hashlib.sha256(open(target, "rb").read()).hexdigest() == expected_sha256:
            return target
        else:
            print(
                f"{target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    response = requests.get(url, stream=True)
    with open(target, "wb") as output:
        size = int(response.headers.get("content-length", 0))
        with tqdm(total=size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=8192):
                output.write(data)
                pbar.update(len(data))

    if hashlib.sha256(open(target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return target


def save_image(image, path):
    if isinstance(image, torch.Tensor):
        image = K.utils.to_pil_image(image)
    image.save(path)


def load_diffusion_model(model_path, device="cpu"):
    model_config = script_util.model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": False,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
    model.requires_grad_(False).eval().to(device)
    if Path(model_path).suffix == ".safetensors":
        safetensors.torch.load_model(model, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    model_wrap = K.external.OpenAIDenoiser(model, diffusion, device=device)
    return model_wrap


def batch_crop(x, out_size, corners, mode="bilinear", padding_mode="zeros"):
    # batch crops out of a single image and resize them all to out_size
    # x, the input image, is NCHW but N must be 1
    # out_size is a tuple, (out_h, out_w)
    # crop corners tensor is N x <0 for h, 1 for w> x <0 for start loc, 1 for end loc>
    n, c, h, w = x.shape
    assert n == 1
    # make base grid, <0 for h, 1 for w> x H x W
    ramp_h = torch.linspace(0, 1, out_size[0], device=x.device)
    ramp_w = torch.linspace(0, 1, out_size[1], device=x.device)
    grid = torch.stack(torch.meshgrid(ramp_h, ramp_w, indexing="ij"), dim=-1)
    # scale corners tensor to the -1 to 1 range used by grid_sample()
    corners = corners / corners.new_tensor([h - 1, w - 1])[None, :, None] * 2 - 1
    # work out the values to scale and shift the h and w grids by
    scales = corners[:, :, 1] - corners[:, :, 0]
    shifts = corners[:, :, 0]
    # scale and shift the grids
    grid = grid[None] * scales[:, None, None, :] + shifts[:, None, None, :]
    # resize and crop
    x = x.expand([corners.shape[0], -1, -1, -1])
    grid = grid.flip(3)
    return F.grid_sample(x, grid, mode, padding_mode, align_corners=False)


class CLIPWrapper(nn.Module):
    def __init__(self, model, preprocess, cutn=32):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.cutn = cutn

    @property
    def cut_size(self):
        return (self.model.visual.input_resolution,) * 2

    @classmethod
    def from_pretrained(cls, clip_name, device="cpu", **kwargs):
        model = clip.load(clip_name)[0].eval().requires_grad_(False).to(device)
        preprocess = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return cls(model, preprocess, **kwargs)

    def encode_text(self, s):
        toks = clip.tokenize(s, truncate=True).to(self.model.logit_scale.device)
        return self.model.encode_text(toks).float()

    def forward(self, x):
        n, c, h, w = x.shape
        max_size = min(w, h)
        size = torch.empty([self.cutn], device=x.device)
        size = nn.init.trunc_normal_(size, 0.8, 0.3, 0.5, 0.95) * max_size
        offsetx = torch.rand([self.cutn], device=x.device) * (w - size)
        offsety = torch.rand([self.cutn], device=x.device) * (h - size)
        offsets = torch.stack([offsety, offsetx], dim=-1)
        corners = torch.stack([offsets, offsets + size[:, None]], dim=-1)
        x = self.preprocess((x + 1) / 2)
        cutouts = batch_crop(x, self.cut_size, corners)
        return self.model.encode_image(cutouts).to(x.dtype)


@torch.no_grad()
def sample_dpm_guided(
    model,
    x,
    sigma_min,
    sigma_max,
    max_h,
    max_cond,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    callback=None,
):
    """DPM-Solver++(1/2/3M) SDE (Kat's splitting version)."""
    noise_sampler = (
        K.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
        if noise_sampler is None
        else noise_sampler
    )
    if solver_type not in {"euler", "midpoint", "heun", "dpm3"}:
        raise ValueError('solver_type must be "euler", "midpoint", "heun", or "dpm3"')

    # Helper functions
    def sigma_to_t(sigma):
        return -torch.log(sigma)

    def t_to_sigma(t):
        return torch.exp(-t)

    def phi_1(x):
        return torch.expm1(-x)

    def h_for_max_cond(t, eta, cond_eps_norm, max_cond):
        # This returns the h that should be used for the given cond_scale norm to keep
        # the norm of its contribution to a step below max_cond at a given t.
        sigma = t_to_sigma(t)
        return (cond_eps_norm / (cond_eps_norm - max_cond / sigma)).log() / (eta + 1)

    # Set up constants
    sigma_min = torch.tensor(sigma_min, device=x.device)
    sigma_max = torch.tensor(sigma_max, device=x.device)
    max_h = torch.tensor(max_h, device=x.device)
    s_in = x.new_ones([x.shape[0]])
    t = sigma_to_t(sigma_max)
    t_end = sigma_to_t(sigma_min)

    # Set up state
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None
    i = 0

    # Main loop
    while t < t_end - 1e-5:
        # Call model and cond_fn
        sigma = t_to_sigma(t)
        denoised, cond_score = model(x, sigma * s_in)

        # Scale step size down if cond_score is too large
        cond_eps_norm = cond_score.mul(sigma).pow(2).mean().sqrt() + 1e-8
        h = h_for_max_cond(t, eta, cond_eps_norm, max_cond)
        h = h.nan_to_num(nan=float("inf")).clamp(max=max_h)
        t_next = torch.minimum(t + h, t_end)
        h = t_next - t
        sigma_next = t_to_sigma(t_next)

        # Callback
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_next": sigma_next,
                    "denoised": denoised,
                }
            )

        # First order step (guided)
        x = (sigma_next / sigma) * torch.exp(-h * eta) * x
        x = x - phi_1(h + eta * h) * (denoised + sigma**2 * cond_score)
        noise = noise_sampler(sigma, sigma_next)
        x = x + noise * sigma_next * phi_1(2 * eta * h * s_noise).neg().sqrt()

        # Higher order correction (not guided)
        if solver_type == "dpm3" and denoised_2 is not None:
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
            d2 = (1 / (r0 + r1)) * (d1_0 - d1_1)
            phi_2 = phi_1(h + eta * h) / (h + eta * h) + 1
            phi_3 = phi_2 / (h + eta * h) - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        elif solver_type in {"heun", "dpm3"} and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = phi_1(h + eta * h) / (h + eta * h) + 1
            x = x + phi_2 * d
        elif solver_type == "midpoint" and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            x = x - 0.5 * phi_1(h + eta * h) * d

        # Update state
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
        t += h
        i += 1

    return x


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", type=str, help="the text prompt")
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="the diffusion model checkpoint to load",
    )
    p.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/16",
        choices=clip.available_models(),
        help="the CLIP model to use",
    )
    p.add_argument(
        "--clip-scale",
        "-cs",
        type=float,
        default=2500.0,
        help="the CLIP guidance scale",
    )
    p.add_argument("--compile", action="store_true", help="torch.compile() the model")
    p.add_argument(
        "--cutn",
        type=int,
        default=32,
        help="the number of random crops to use per step",
    )
    p.add_argument("--device", type=str, default=None, help="the device to use")
    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="the multiplier for the noise variance. 0 gives ODE sampling, 1 gives standard diffusion SDE sampling.",
    )
    p.add_argument(
        "--max-cond",
        type=float,
        default=0.05,
        help="the maximum amount that guidance is allowed to perturb a step",
    )
    p.add_argument(
        "--max-h",
        type=float,
        default=0.1,
        help="the maximum step size",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=Path("out.png"), help="the output file"
    )
    p.add_argument(
        "--save-all", action="store_true", help="save all intermediate denoised images"
    )
    p.add_argument(
        "--size", type=int, nargs=2, default=(512, 512), help="the output size"
    )
    p.add_argument(
        "--solver",
        type=str,
        choices=("euler", "midpoint", "heun", "dpm3"),
        default="dpm3",
        help="the SDE solver type",
    )
    p.add_argument("--seed", type=int, default=0, help="the random seed")
    args = p.parse_args()

    print(Panel(Align("CLIP Guided Diffusion", "center")))

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f'Using device: "{device}"')

    torch.set_float32_matmul_precision("high")

    # Load diffusion model
    print("Loading diffusion model...")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = download_file(
            url="https://models.rivershavewings.workers.dev/512x512_diffusion_uncond_finetune_008100.safetensors",
            root=Path(torch.hub.get_dir()) / "checkpoints" / "rivershavewings",
            expected_sha256="02e212cbec7c9012eb12cd63fef6fa97640b4e8fcd6c6e1f410a52eea1925fe1",
        )
    model = load_diffusion_model(checkpoint, device=device)
    sigma_min, sigma_max = model.sigmas[0].item(), model.sigmas[-1].item()
    size_fac = (args.size[0] * args.size[1]) / (512 * 512)

    # Load CLIP and encode prompt
    print("Loading CLIP...")
    clip_wrap = CLIPWrapper.from_pretrained(
        args.clip_model, device=device, cutn=args.cutn
    )
    target = clip_wrap.encode_text(args.prompt)

    # Wrap the model in a function that also computes and returns the cond_score
    def cond_model(x, sigma, **kwargs):
        denoised = None

        def loss_fn(x):
            nonlocal denoised
            denoised = model(x, sigma, **kwargs)
            image_embeds = clip_wrap(denoised)
            clip_score = torch.cosine_similarity(image_embeds, target, dim=-1).mean()
            loss = -clip_score * args.clip_scale * size_fac
            return loss

        grad = torch.autograd.functional.vjp(loss_fn, x)[1]
        return denoised, -grad

    if args.compile:
        cond_model = torch.compile(cond_model)

    # Set up callback
    class Callback:
        def __enter__(self):
            self.ex = futures.ThreadPoolExecutor()
            self.pbar = tqdm()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.pbar.close()
            self.ex.shutdown()

        def __call__(self, info):
            self.pbar.update(1)
            i = info["i"]
            sigma = info["sigma"].item()
            sigma_next = info["sigma_next"].item()
            h = math.log(sigma / sigma_next)
            print(f"step {i}, sigma: {sigma:g}, h: {h:g}")
            if args.save_all:
                path = args.output.with_stem(args.output.stem + f"_{i:05}")
                self.ex.submit(save_image, info["denoised"][0], path)

    with Callback() as cb:
        # Draw random noise
        torch.manual_seed(args.seed)
        x = torch.randn([1, 3, args.size[1], args.size[0]], device=device) * sigma_max
        ns = K.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)

        # Sample
        print("Sampling...")
        try:
            samples = sample_dpm_guided(
                model=cond_model,
                x=x,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                max_h=args.max_h,
                max_cond=args.max_cond,
                eta=args.eta,
                noise_sampler=ns,
                solver_type=args.solver,
                callback=cb,
            )

            # Save the image
            print(f"Saving to {args.output}...")
            save_image(samples[0], args.output)
        except KeyboardInterrupt:
            print("Interrupted")


if __name__ == "__main__":
    main()
