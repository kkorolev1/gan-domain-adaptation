import os
import torch
import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm, trange
from models.stylegan2.model import Generator
from PIL import Image
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from argparse import Namespace
from utils.common import tensor2im
from models.psp import pSp
from degan.datasets import FFHQDataset
from degan.utils import get_concat_h


def load_generator(config):
    g_ema = Generator(config["size"], config["style_dim"], config["n_mlp"])
    g_ema.load_state_dict(torch.load(config["ckpt"])["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    print("Generator successfully loaded")
    return g_ema

def load_psp(config):
    ckpt = torch.load(config["ckpt"], map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = config["ckpt"]
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print("pSp successfully loaded")
    return net

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

@torch.no_grad()
def get_latents(net, dataset):
    latents_lst = []
    for i in trange(len(dataset)):
        _, latents = run_on_batch(dataset[i].unsqueeze(0), net)
        latents_lst.append(latents)
    return latents_lst

@torch.no_grad()
def run_g_on_latents(g_ema, latents_lst, direction, target_img, save_dir, alpha=1):
    for i, source in enumerate(tqdm(latents_lst)):
        direction = direction.cuda()
        source = source.cuda()
        source_img, _ = g_ema([source], input_is_latent=True, randomize_noise=False)
        source_amp, _ = g_ema([source + direction * alpha], input_is_latent=True,
                                randomize_noise=False)
        source_img = tensor2im(source_img.squeeze(0))
        stylized_img = tensor2im(source_amp.squeeze(0))
        result_img = get_concat_h(get_concat_h(source_img, target_img), stylized_img)
        result_img.save(os.path.join(save_dir, f"{i + 1}.png"))

@hydra.main(version_base=None, config_path="../../configs", config_name="target_clip")
def main(config):
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)
    root_path = config["root_path"]
    
    generator = load_generator(config["generator"])
    encoder = load_psp(config["encoder"])

    dataset = FFHQDataset(config["ffhq_path"], 256)
    print(f"Dataset size {len(dataset)}")

    latents_lst = get_latents(encoder, dataset)
    eval_targets = config["eval_targets"]

    with torch.no_grad():
        for target_name in eval_targets:
            print(f"Processing {target_name} target")
            target_img = Image.open(os.path.join(root_path, config["targets"][target_name])).convert("RGB").resize((1024, 1024))
            direction = torch.from_numpy(np.load(os.path.join(root_path, config["dirs"][target_name])))
            save_dir = os.path.join(config["save_dir"], target_name)
            os.makedirs(save_dir, exist_ok=True)
            run_g_on_latents(generator, latents_lst, direction, target_img, save_dir)
            

if __name__ == "__main__":
    main()