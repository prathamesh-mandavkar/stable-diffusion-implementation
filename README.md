# Stable Diffusion Implementation in PyTorch

This repository provides a from-scratch implementation of **Stable Diffusion** in PyTorch. The implementation draws from the official research paper and educational resources such as [Umar Jamil's tutorial](https://youtu.be/ZBKpAp_6TGI?si=f9OcG-hBk8z8TK6w). It is intended to serve as a clear and modular reference for understanding and experimenting with diffusion-based generative models.

## Overview

Stable Diffusion is a latent text-to-image diffusion model capable of generating high-quality images from textual prompts. This repository breaks down the architecture into well-defined components including the encoder, decoder, denoising model (U-Net), attention mechanisms, and inference pipeline.

## Features

- Modular PyTorch implementation of Stable Diffusion components
- Integration of CLIP for text-conditioning
- VAE-based latent space modeling
- DDPM-based forward and reverse diffusion processes
- Inference pipeline for prompt-to-image generation
- Example notebooks for training and demonstration

## Project Structure

```
.
├── add_noise.ipynb         # Noise visualization notebook
├── attention.py            # Attention mechanisms
├── clip.py                 # CLIP text encoder integration
├── ddpm.py                 # Diffusion model implementation
├── decoder.py              # VAE decoder module
├── demo.ipynb              # Image generation demo notebook
├── diffusion.py            # Forward and reverse diffusion processes
├── encoder.py              # VAE encoder module
├── model_converter.py      # Model format conversion utilities
├── model_loader.py         # Load and prepare pretrained models
├── pipeline.py             # Inference pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup and Usage

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Download Required Files

Create a `data/` directory and download the following assets:

1. **Tokenizer files** (from Hugging Face):
   - [vocab.json](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/tokenizer/vocab.json)
   - [merges.txt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/tokenizer/merges.txt)

2. **Model checkpoint**:
   - [v1-5-pruned-emaonly.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt)

3. **Optional fine-tuned checkpoints** (compatible with this implementation):
   - [InkPunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main)
   - [Illustration Diffusion (Hollie Mengert)](https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main)

### Run the Demo

Open `demo.ipynb` to run the image generation pipeline using your downloaded assets.

## References

This project is informed by the following works and repositories:

- [Stable Diffusion Official Repository (CompVis)](https://github.com/CompVis/stable-diffusion)
- [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers)
- [Stable Diffusion TensorFlow (divamgupta)](https://github.com/divamgupta/stable-diffusion-tensorflow)
- [PyTorch Implementation by kjsman](https://github.com/kjsman/stable-diffusion-pytorch)

## Acknowledgments

Credit to [Umar Jamil](https://www.youtube.com/@UmarJamil) for the detailed walkthrough on Stable Diffusion implementation, which significantly inspired this project.
