# Development of domain adaptation methods for generative models
Faculty of Computer Science, Higher School of Economics

**Scientific advisors**: Aibek Alanov, Maksim Nakhodnov

## Abstract
Modern generative adversarial networks (GANs) enable the synthesis of high-quality images
and provide tools for fine-grained image manipulation. However, out-of-domain generation requires
an additional fine-tuning of a generator or non-flexible latent optimization, which requires training
for each new image. A novel encoder, which learns offsets in a S space and allows a GAN generator
to adapt to a new domain in a single pass, is proposed and implemented. An ablation study of loss
components and usage of projection embeddings is conducted. A method for regularization in a
multi-domain setting is proposed. A thorough comparison with prior domain adaptation methods
is made.

<p align="center">
  <img src="https://i.ibb.co/rtcj69D/Template-00000.png" border="0" />
</p>

## Approach
In this work we focus on an encoder-based approach, inspired by BlendGAN, for imaged-based domain adaptation. Similarly, our task is to train an encoder, which takes a domain image and returns it's representation that can be incorporated within a generator to synthesize adapted images to a new domain. We call this approach as <b>D</b>omain <b>E</b>ncoder <B>GAN</b> (DEGAN). Crucial aspect is that neither generator, nor discriminator are trained within this pipeline, making a training process easier and more flexible. In our method an encoder predicts offsets in a StyleSpace, taking into account the results from StyleDomain paper, which show that parameterization in a StyleSpace is compact and efficient comparing to others. Generally, our optimization process looks like

$$
\mathcal{L}(\phi, G, (I_d)) \to \min_{\phi}
$$

where $\phi$ is a domain encoder, $G$ is a frozen pretrained StyleGAN2 generator, $(I_d)$ is a set of domain images.

Our training pipeline looks as follows:

1) Sample a latent $z \sim \mathcal{N}(0, I)$ and propagate it through a mapping network and affine layers to get a style vector $s \in \mathcal{S}$, which plays role of representation of a source image.
2) Given a domain image $I_d$, predict a domain offset in a StyleSpace using a domain encoder $\phi(I_d) = \Delta s$.
3) Generate a source image $I_s = G(s)$ and an adapted image $I_g = G(s + \Delta s)$ using a frozen generator.
4) Get embeddings $E_{CLIP}(I_s), E_{CLIP}(I_d), E_{CLIP}(I_g)$ using a pretrained ViT-B/16 CLIP.
5) Based on the objects above, calculate a loss function and make a step of an optimizer.

Following latest achievements in computer vision, we choose ViT-B/16 as a backbone for a domain encoder. Because we want to predict offsets in a StyleSpace of dimensionality 9088, there will be a bottleneck, if we simply train a head above the [CLS] embedding of dimensionality 768 after the last encoder layer of ViT. Furthermore, different layers of ViT operate on different image resolutions. That coincides with the structure of a StyleSpace, that consists of vectors, which are utilized on 9 groups of resolutions from 4x4 to 1024x1024. These facts motivated us to add trainable heads on 9 last layers of a backbone, such that the predictions for a particular resolution are consistent with the resolution of an image being processed. StyleGAN2 generator uses two convolution and one ToRGB layers on each resolution, except for 4x4. Therefore, each head takes a [CLS] embedding from the current layer and propagates it through several independent 3-layer MLPs with LayerNorm and GeLU activation to predict an offset for a particular layer in a generator. Our domain encoder has about 37 million parameters. A detailed version of an architecture of a domain encoder is presented in figure below.
<p align="center">
  <img src="https://i.ibb.co/QnbTH2F/Model-Arch-00000-00000.png" alt="Model-Arch-00000-00000" border="0" />
</p>


## Training
Training config can be found in `configs`.
```shell
python train.py
```
