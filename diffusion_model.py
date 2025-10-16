# This file is a concatenation of DiffPIR codes available here: https://github.com/yuanzhi-zhu/DiffPIR/tree/main
# This code is taken (with minor modifications) from https://github.com/yuanzhi-zhu/DiffPIR/tree/main

import torch
from abc import abstractmethod
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
from torch import Tensor
from torch.nn import Linear, GroupNorm
from typing import Union

#Base
class Denoiser(torch.nn.Module):
    r"""
    Base class for denoiser models.

    Provides a template for defining denoiser models.

    While most denoisers :math:`\denoisername` are designed to handle Gaussian noise
    with variance :math:`\sigma^2`, this is not mandatory.

    .. note::

        A Denoiser can be converted into a :class:`Reconstructor <deepinv.models.Reconstructor>`
        by using the :class:`deepinv.models.ArtifactRemoval` class.

    The base class inherits from :class:`torch.nn.Module`.

    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, x, sigma, **kwargs):
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.

        :param torch.Tensor x: noisy input.
        :param torch.Tensor, float sigma: noise level.
        :returns: (:class:`torch.Tensor`) Denoised tensor.
        """
        return NotImplementedError

    @staticmethod
    def _handle_sigma(
        sigma: Union[float, torch.Tensor],
        batch_size: int = None,
        ndim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwarg,
    ):
        r"""
        Convert various noise level types to the appropriate format for batch processing.
            If `sigma` is a single float or int, the same value will be used for each sample in the batch.
            If `sigma` is a tensor, it should be of shape `(batch_size,)` or a scalar.
            If `sigma` is a list, it should be of length `batch_size` or `1`.

        To be overridden by subclasses if necessary.

        :param float, torch.Tensor sigma: noise level.
        :param int batch_size: number of samples in the batch (optional).
        :param int ndim: number of dimensions of the input tensor (optional).
        :param torch.device device: device to which the tensor should be moved (optional).
        :param torch.dtype dtype: data type of the tensor (optional).
        :param args: additional positional arguments.
        :param kwarg: additional keyword arguments.

        :returns: noise levels for each sample in the batch adapted to the denoiser.
        """
        if isinstance(sigma, (float, int)):
            sigma = float(sigma)
        elif isinstance(sigma, torch.Tensor):
            sigma = sigma.squeeze().to(dtype=dtype, device=device)
        elif isinstance(sigma, list):
            sigma = torch.tensor(sigma, dtype=dtype, device=device).squeeze()
        elif isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma, dtype=dtype, device=device).squeeze()
        else:
            raise TypeError(
                f"Sigma must be a float, int, or torch.Tensor. Got {type(sigma)}."
            )

        # Will reshape to (batch_size,) if batch_size is not None
        if batch_size is not None:
            # duplicate sigma for each sample in the batch
            if isinstance(sigma, float):
                sigma = torch.tensor([sigma] * batch_size, dtype=dtype, device=device)
            elif sigma.ndim == 0:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) == 1:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) != batch_size:
                raise ValueError(
                    f"Sigma tensor size {sigma.size(0)} does not match batch size {batch_size}."
                )

        # Will reshape to (batch_size, 1, ..., 1) if ndim is not None
        if ndim is not None:
            if isinstance(sigma, float):
                sigma = torch.tensor(sigma, dtype=dtype, device=device).view(
                    1, *([1] * (ndim - 1))
                )
            elif sigma.ndim == 0:
                sigma = sigma.view(1, *([1] * (ndim - 1)))
            elif sigma.ndim == 1:
                sigma = sigma.view(-1, *([1] * (ndim - 1)))
            else:
                raise ValueError(
                    f"Sigma tensor has {sigma.ndim} dimensions, expected 0 or 1."
                )
        return sigma


class Reconstructor(torch.nn.Module):
    r"""
    Base class for reconstruction models.

    Provides a template for defining reconstruction models.

    Reconstructors provide a signal estimate ``x_hat`` as ``x_hat = model(y, physics)`` where ``y`` are the measurements
    and ``physics`` is the forward model :math:`A` (possibly including information about the noise distribution too).

    The base class inherits from :class:`torch.nn.Module`.

    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, y, physics, **kwargs):
        r"""
        Applies reconstruction model :math:`\inversef{y}{A}`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model :math:`A`.
        :returns: (:class:`torch.Tensor`) reconstructed tensor.
        """
        return NotImplementedError

# Utils
def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def zip_strict(*iterables):
    # This was a missing helper function.
    if not iterables:
        return
    length = len(iterables[0])
    for iterable in iterables[1:]:
        if len(iterable) != length:
            raise ValueError("Iterables have different lengths")
    return zip(*iterables)

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)


def get_weights_url(model_name, file_name):
    return (
        "https://huggingface.co/deepinv/"
        + model_name
        + "/resolve/main/"
        + file_name
        + "?download=true"
    )


def test_pad(model, L, modulo=16):
    """
    Pads the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
    """
    h, w = L.size()[-2:]
    padding_bottom = int(np.ceil(h / modulo) * modulo - h)
    padding_right = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


def test_onesplit(model, L, refield=32, sf=1):
    """
    Changes the size of the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models.

    :param model: model.
    :param L: input Low-quality image.
    :param refield: effective receptive field of the network, 32 is enough.
    :param sf: scale factor for super-resolution, otherwise 1.
    """
    h, w = L.size()[-2:]
    top = slice(0, (h // 2 // refield + 1) * refield)
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)
    left = slice(0, (w // 2 // refield + 1) * refield)
    right = slice(w - (w // 2 // refield + 1) * refield, w)
    Ls = [
        L[..., top, left],
        L[..., top, right],
        L[..., bottom, left],
        L[..., bottom, right],
    ]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., : h // 2 * sf, : w // 2 * sf] = Es[0][..., : h // 2 * sf, : w // 2 * sf]
    E[..., : h // 2 * sf, w // 2 * sf : w * sf] = Es[1][
        ..., : h // 2 * sf, (-w + w // 2) * sf :
    ]
    E[..., h // 2 * sf : h * sf, : w // 2 * sf] = Es[2][
        ..., (-h + h // 2) * sf :, : w // 2 * sf
    ]
    E[..., h // 2 * sf : h * sf, w // 2 * sf : w * sf] = Es[3][
        ..., (-h + h // 2) * sf :, (-w + w // 2) * sf :
    ]
    return E


# Basic blocks for defining the architecture of the models
# Adapted from https://github.com/NVlabs/edm

"""
Model architectures and preconditioning schemes used in the paper
Elucidating the Design Space of Diffusion-Based Generative Models: https://arxiv.org/pdf/2206.00364

"""

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class UpDownConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.outer(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, num_groups=32, eps=eps)
        self.conv0 = UpDownConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
        )
        self.norm1 = GroupNorm(num_channels=out_channels, num_groups=32, eps=eps)
        self.conv1 = UpDownConv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = UpDownConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, num_groups=32, eps=eps)
            self.qkv = UpDownConv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = UpDownConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self, num_channels: int, max_positions: int = 10000, endpoint: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: Tensor):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, scale: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x: Tensor):
        x = x.outer((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# Model Architecture
class DiffUNet(Denoiser):
    r"""
    Diffusion UNet model.

    This is the model with attention and timestep embeddings from `Ho et al. <https://arxiv.org/abs/2108.02938>`_;
    code is adapted from https://github.com/jychoi118/ilvr_adm.

    It is possible to choose the `standard model <https://arxiv.org/abs/2108.02938>`_
    with 128 hidden channels per layer (trained on FFHQ)
    and a `larger model <https://arxiv.org/abs/2105.05233>`_ with 256 hidden channels per layer (trained on ImageNet128)

    A pretrained network for (in_channels=out_channels=3)
    can be downloaded via setting ``pretrained='download'``.

    The network can handle images of size :math:`2^{n_1}\times 2^{n_2}` with :math:`n_1,n_2 \geq 5`.

    .. warning::

        This model has 2 forward modes:

        * ``forward_diffuse``: in the first mode, the model takes a noisy image and a timestep as input and estimates the noise map in the input image. This mode is consistent with the original implementation from the authors, i.e. it assumes the same image normalization.
        * ``forward_denoise``: in the second mode, the model takes a noisy image and a noise level as input and estimates the noiseless underlying image in the input image. In this case, we assume that images have values in [0, 1] and a rescaling is performed under the hood.


    :param int in_channels: channels in the input Tensor.
    :param int out_channels: channels in the output Tensor.
    :param bool large_model: if True, use the large model with 256 hidden channels per layer trained on ImageNet128
        (weights size: 2.1 GB).
        Otherwise, use a smaller model with 128 hidden channels per layer trained on FFHQ (weights size: 357 MB).
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at
        random using Pytorch's default initialization.
        If ``pretrained='download'``, the weights will be downloaded from an online repository
        (only available for 3 input and output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        large_model=False,
        use_fp16=False,
        pretrained="download",
        num_classes=None,
    ):
        super().__init__()

        if large_model:
            model_channels = 256
            num_res_blocks = 2
            attention_resolutions = "8,16,32"
        else:
            model_channels = 128
            num_res_blocks = 1
            attention_resolutions = "16"

        dropout = 0.1
        conv_resample = True
        dims = 2
        use_checkpoint = False
        num_heads = 4
        num_head_channels = 64
        num_heads_upsample = -1
        use_scale_shift_norm = True
        resblock_updown = True
        use_new_attention_order = False

        out_channels = 6 if out_channels == 3 else out_channels
        channel_mult = (1, 1, 2, 2, 4, 4)

        img_size = 256
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size // int(res))
        attention_resolutions = tuple(attention_ds)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.img_size = img_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes+1, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 3 and out_channels == 6 and not large_model:
                    name = "diffusion_ffhq_10m.pt"
                elif in_channels == 3 and out_channels == 6 and large_model:
                    name = "diffusion_openai.pt"
                else:
                    raise ValueError(
                        "no existing pretrained model matches the requested configuration"
                    )
                url = get_weights_url(model_name="diffunet", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)

            self.load_state_dict(ckpt, strict=True)
            self.eval()

    def forward(self, x, t, y=None, type_t="noise_level"):
        r"""
        Apply the model to an input batch.

        This function takes a noisy image and either a timestep or a noise level as input. Depending on the nature of
        ``t``, the model returns either a noise map (if ``type_t='timestep'``) or a denoised image (if
        ``type_t='noise_level'``).

        :param torch.Tensor x: an `(N, C, ...)` Tensor of inputs.
        :param torch.Tensor t: a 1-D batch of timesteps or noise levels.
        :param torch.Tensor y: an (N) Tensor of labels, if class-conditional. Default=None.
        :param str type_t: Nature of the embedding `t`. In traditional diffusion model, and in the authors' code, `t` is
                       a timestep linked to a noise level; in this case, set ``type_t='timestep'``. We can also choose
                       ``t`` to be a noise level directly and use the model as a denoiser; in this case, set
                       ``type_t='noise_level'``. Default: ``'timestep'``.
        :return: an `(N, C, ...)` Tensor of outputs. Either a noise map (if ``type_t='timestep'``) or a denoised image
                    (if ``type_t='noise_level'``).
        """
        if x.shape[-2] < 520 and x.shape[-1] < 520:
            pad = (-x.size(-1) % 32, 0, -x.size(-2) % 32, 0)
            x = torch.nn.functional.pad(x, pad, mode="circular")
            if type_t == "timestep":
                out = self.forward_diffusion(x, t, y=y)
            elif type_t == "noise_level":
                out = self.forward_denoise(x, t, y=y)
            else:
                raise ValueError('type_t must be either "timestep" or "noise_level"')
            return out[..., pad[-2] :, pad[-4] :]
        else:
            return self.patch_forward(x, t, y=y, type_t=type_t, patch_size=512)

    def patch_forward(self, x, t, y=None, type_t="noise_level", patch_size=512):
        r"""
        Splits an image tensor into patches (without overlapping), applies the model to each patch, and reconstructs the full image.

        :param x: Input low-quality image tensor of shape (B, C, H, W).
        :param patch_size: Size of the patches to split into.
        :param \*args: Additional positional arguments for the model.
        :param \*\*kwargs: Additional keyword arguments for the model.

        :return: Reconstructed image tensor.
        """

        pad_input = (-x.size(-1) % patch_size, 0, -x.size(-2) % patch_size, 0)
        x = torch.nn.functional.pad(x, pad_input, mode="circular")

        B, C, H, W = x.shape

        # Calculate number of patches needed
        h_patches = int((H + patch_size - 1) // patch_size)  # Ceiling division
        w_patches = int((W + patch_size - 1) // patch_size)

        # Pad image to fit exactly into patches if necessary
        pad_h = int(h_patches * patch_size - H)
        pad_w = int(w_patches * patch_size - W)
        x_padded = F.pad(x, (pad_h, 0, pad_w, 0), mode="circular")

        # Process patches
        E_padded = torch.zeros(B, C, H + pad_h, W + pad_w).type_as(x)

        for i in range(h_patches):
            for j in range(w_patches):
                h_start = int(i * patch_size)
                w_start = int(j * patch_size)
                patch = x_padded[
                    ..., h_start : h_start + patch_size, w_start : w_start + patch_size
                ]

                # Apply model to the patch
                E_patch = self.forward(patch, t, y=y, type_t=type_t)

                # Place processed patch in the output tensor
                E_padded[
                    ...,
                    h_start : (h_start + patch_size),
                    w_start : (w_start + patch_size),
                ] = E_patch

        # Crop back to original size
        E = E_padded[..., :H, :W]

        return E[..., pad_input[-2] :, pad_input[-4] :]

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward_diffusion(self, x, timesteps, y=None):
        r"""
        Apply the model to an input batch.

        This function takes a noisy image and a timestep as input (and not a noise level) and estimates the noise map
        in the input image.
        The image is assumed to be in range [-1, 1] and to have dimensions with width and height divisible by a
        power of 2.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional. Default=None.
        :return: an `(N, C, ...)` Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_alpha_prod(
        self, beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000
    ):
        """
        Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.
        """
        betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=torch.float32
        )
        # .to(self.device) Removing this for now, can be done outside
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        return (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        )

    def find_nearest(self, array, value):
        """
        Find the argmin of the nearest value in a tensor.
        """
        idx = (torch.abs(array[:, None] - value[None, :])).argmin(dim=0)
        return idx

    def forward_denoise(self, x, sigma, y=None):
        r"""
        Applies the denoising model to an input batch.

        This function takes a noisy image and a noise level as input (and not a timestep) and estimates the noiseless
        underlying image in the input image.
        The input image is assumed to be in range [0, 1] (up to noise) and to have dimensions with width and height
        divisible by a power of 2.

        .. note::
            The DiffUNet assumes that images are scaled as :math:`\sqrt{\alpha_t} x + (1-\alpha_t) \epsilon`
            thus an additional rescaling by :math:`\sqrt{\alpha_t}` is performed within this function, along with
            a mean shift by correction by :math:`0.5 - \sqrt{\alpha_t} 0.5`.

        :param torch.Tensor x: an `(N, C, ...)` Tensor of inputs.
        :param torch.Tensor sigma: a 1-D batch of noise levels.
        :param torch.Tensor y: an (N) Tensor of labels, if class-conditional. Default=None.
        :return: an `(N, C, ...)` Tensor of outputs.
        """

        sigma = self._handle_sigma(
            sigma, batch_size=x.size(0), ndim=x.ndim, device=x.device, dtype=x.dtype
        )
        alpha = 1 / (1 + 4 * sigma**2)
        x = alpha.sqrt() * (2 * x - 1)
        sigma = sigma * alpha.sqrt()
        (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        ) = self.get_alpha_prod()

        timesteps = self.find_nearest(
            sqrt_1m_alphas_cumprod.to(x.device), sigma.squeeze(dim=(1, 2, 3)) * 2
        )  # Factor 2 because image rescaled in [-1, 1]

        timesteps = timesteps.to(x.device)
        noise_est_sample_var = self.forward_diffusion(x, timesteps, y=y)
        noise_est = noise_est_sample_var[:, :3, ...]
        denoised = (x - noise_est * sigma * 2) / sqrt_alphas_cumprod.to(x.device)[
            timesteps
        ].view(-1, 1, 1, 1)
        denoised = denoised.clamp(-1, 1)
        return (denoised + 1) / 2


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where `forward()` takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param torch.Tensor x: an `(N, C, ...)` Tensor of features.
        :param torch.Tensor emb: an (N x emb_channels) Tensor of timestep embeddings.
        :return: an `(N, C, ...)` :class:`torch.Tensor` of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()