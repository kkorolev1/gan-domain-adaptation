from tqdm import tqdm
import os

import cv2
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from model import Generator
from psp_encoder.psp_encoders import PSPEncoder
from utils import ten2cv, cv2ten
import glob
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="../../configs", config_name="blendgan")
def main(config):
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)

    device = 'cuda'

    g_config = config["generator"]

    checkpoint = torch.load(config["pretrained"]["blendgan"])
    model_dict = checkpoint['g_ema']
    print('ckpt: ', g_config["ckpt"])

    g_ema = Generator(
        g_config["size"], g_config["style_dim"], g_config["n_mlp"], load_pretrained_vgg=False
    ).to(device)
    g_ema.load_state_dict(model_dict)
    g_ema.eval()

    psp_encoder = PSPEncoder(config["pretrained"]["e4e"], output_size=g_config["size"]).to(device)
    psp_encoder.eval()

    input_img_paths = sorted(glob.glob(os.path.join(config["input_img_path"], '*.*')))
    style_img_paths = sorted(glob.glob(os.path.join(config["style_img_path"], '*.*')))[:]

    num = 0

    for input_img_path in tqdm(input_img_paths):
        num += 1

        name_in = os.path.splitext(os.path.basename(input_img_path))[0]
        img_in = cv2.imread(input_img_path, 1)
        img_in_ten = cv2ten(img_in, device)
        img_in = cv2.resize(img_in, (g_config["size"], g_config["size"]))

        for style_img_path in style_img_paths:
            name_style = os.path.splitext(os.path.basename(style_img_path))[0]
            img_style = cv2.imread(style_img_path, 1)
            img_style_ten = cv2ten(img_style, device)
            img_style = cv2.resize(img_style, (g_config["size"], g_config["size"]))

            with torch.no_grad():
                sample_style = g_ema.get_z_embed(img_style_ten)
                sample_in = psp_encoder(img_in_ten)
                img_out_ten, _ = g_ema([sample_in], z_embed=sample_style, add_weight_index=6,
                                       input_is_latent=True, return_latents=False, randomize_noise=False)
                img_out = ten2cv(img_out_ten)
            out = np.concatenate([img_in, img_style, img_out], axis=1)
            # out = img_out
            save_dir = os.path.join(config["save_dir"], name_style)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'{name_in}.png'), out)


if __name__ == '__main__':
    main()
