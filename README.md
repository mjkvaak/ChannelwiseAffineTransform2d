## Getting started

Get [Miniconda](https://repo.anaconda.com/miniconda/) and install prerequisites

```sh
conda create -y -n affine python=3.10 pip jupyter
conda activate affine
pip install einops matplotlib tqdm

```
Follow the [PyTorch documentation](https://pytorch.org/get-started/locally/) to install `torch`, e.g. 
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## How to use
The `ChannelwiseAffineTransform2D` module derives from `nn.Module` so it can be used as any other PyTorch module. The module supports an arbitrary number of channels. 

Example usage:
```
model = ChannelwiseAffineTransform2D(num_channels=...)
```
