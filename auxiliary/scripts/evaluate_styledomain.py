import os
import torch
import PIL
from tqdm import trange, tqdm
from argparse import Namespace

from torchvision.transforms import Resize
from omegaconf import OmegaConf
from hydra.utils import instantiate
import hydra
from core.utils.example_utils import (
    Inferencer, to_im
)
from restyle_encoders.e4e import e4e
from core.utils.reading_weights import read_weights

from pathlib import Path
from collections import defaultdict

from examples.draw_util import weights, IdentityEditor, StyleEditor


def get_concat_h(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


@hydra.main(version_base=None, config_path="../../configs", config_name="styledomain")
def main(config):
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)

    device = "cuda"
    dataset = instantiate(config["ffhq_test"])
    print(f"Dataset size {len(dataset)}")

    model_path = config['pretrained']['e4e']
    ckpt = torch.load(model_path, map_location='cuda:0')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = e4e(opts).eval().to(device)

    w_singles = []

    with torch.no_grad():
        for i in trange(len(dataset)):
            img = dataset[i].unsqueeze(0).to(device)
            _, w_plus = net(img, randomize_noise=False, return_latents=True)
            w_singles.append(w_plus)

    row_domains = config["eval_targets"]
    style_to_editor = {
        d: StyleEditor(read_weights(weights[d])) if d != 'original' else IdentityEditor() for d in row_domains
    }

    offset_pow = 0.85

    dom_to_pow = defaultdict(lambda : offset_pow, {
        'original': 0.,
        'sketch': 0.7,
        'anastasia': 0.7,
        'pixar': 0.75,
        'botero': 0.75,
        'joker': 0.65,
        'edvard_munch_painting': 0.95,
        'modigliani_painting': 0.75
    })

    ckpt = read_weights(weights["anastasia"])
    ckpt_ffhq = {'sg2_params': ckpt['sg2_params']}
    ckpt_ffhq['sg2_params']['checkpoint_path'] = weights["ffhq"]
    model = Inferencer(ckpt_ffhq, device).to(device)
    resize = Resize(1024)

    s_singles = [model.sg2_source.get_s_code([w_single], input_is_latent=True) for w_single in w_singles]

    for row_domain in row_domains:
        im_style = PIL.Image.open(os.path.join(config["domains_dir"], f"{row_domain}.png"))
        save_dir = os.path.join(config["save_dir"], row_domain)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Processing {row_domain}")
        for i, s_single in enumerate(tqdm(s_singles)):
            model.sg2_source.generator.load_state_dict(torch.load(weights["ffhq"])['g_ema'])
            s_edited = style_to_editor[row_domain](s_single, power=dom_to_pow[row_domain])
            src_im, _ = model.sg2_source(
                s_edited, is_s_code=True
            )
            im_stylized = to_im(resize(src_im))
            src_im, _ = model.sg2_source(
                s_single, is_s_code=True
            )
            im_source = to_im(resize(src_im))
            im = get_concat_h(get_concat_h(im_source, im_style), im_stylized)
            im.save(os.path.join(save_dir, f"{i+1}.png"))

if __name__ == '__main__':
    main()
