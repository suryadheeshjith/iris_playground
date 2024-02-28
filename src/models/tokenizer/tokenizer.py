"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class BaseTokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim) # Embedding module containing vocab_size tensors of size embed_dim (vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
    
    def __repr__(self) -> str:
        return "base_tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach() # https://ai.stackexchange.com/questions/26770/in-vq-vae-code-what-does-this-line-of-code-signify
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        return y.add(1).div(2)
    
    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        raise NotImplementedError
    
    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        raise NotImplementedError
    
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        raise NotImplementedError

class Tokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        super().__init__(vocab_size, embed_dim)
        # self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        # self.embedding = nn.Embedding(vocab_size, embed_dim) # Embedding module containing vocab_size tensors of size embed_dim (vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        # self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

    def __repr__(self) -> str:
        return "tokenizer"

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean() # Slow down encoder outs so that codebook does not go dimensionless, https://stats.stackexchange.com/questions/595049/whats-the-role-of-the-commitment-loss-in-vq-vae

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W) doesnt necessarily have to be N because can also be N x T (see compute loss)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x) # (..., z_channels, H', W') = (..., 512, 4, 4)
        z = self.pre_quant_conv(z) # (..., embed_dim, H', W') = (..., 512, 4, 4)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e') # (... x H' x W', embed_dim) = (... x 16, 512)
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t()) # (... x H' x W', emb_dim), (... x 16, 512) = (... x 16, 1) + (... x 16, 1) -  (... x 16, 512)
        tokens = dist_to_embeddings.argmin(dim=-1) # (... x H' x W'), token for each pixel in embedded space (..., embed_dim, H', W') (16)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous() # get embedding for each token and reshape (..., emb_dim, H', W') (..., 512, 4, 4)

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:]) # (..., emb_dim, H', W') (..., 512, 4, 4)
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:]) # (..., emb_dim, H', W') (..., 512, 4, 4)
        tokens = tokens.reshape(*shape[:-3], -1) # (..., H' x W') (..., 16)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:]) # (..., emb_dim, H', W') (..., 512, 4, 4)
        z_q = self.post_quant_conv(z_q) # (..., emb_dim, H', W') (..., 512, 4, 4)
        rec = self.decoder(z_q) # (..., C, H, W) (..., 3, 64, 64)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:]) # (..., C, H, W) (..., 3, 64, 64)
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec


class Tokenizer_RAM(BaseTokenizer):
    def __init__(self, vocab_size, embed_dim, in_channels, hidden_size, out_channels) -> None:
        super().__init__(vocab_size, embed_dim)
        assert in_channels == out_channels
        self.vocab_size = vocab_size
        self.pre_quant_conv = torch.nn.Conv2d(in_channels, embed_dim, 1)

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim) # Embedding module containing vocab_size tensors of size embed_dim (vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, out_channels, 1)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels),
        )

        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)


    def __repr__(self) -> str:
        return "tokenizer_ram"

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c -> (b t) c'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean() # Slow down encoder outs so that codebook does not go dimensionless, https://stats.stackexchange.com/questions/595049/whats-the-role-of-the-commitment-loss-in-vq-vae

        reconstruction_loss = torch.abs(observations - reconstructions).mean()

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C) doesnt necessarily have to be N because can also be N x T (see compute loss)
        x = x.view(-1, shape[-1])
        z = self.encoder(x) # (..., emb_dim)
        dist_to_embeddings = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t()) # (..., emb_dim), (..., emb_dim) = (..., 1) + (..., 1) -  (..., emb_dim)
        tokens = dist_to_embeddings.argmin(dim=-1) # (...)
        z_q = self.embedding(tokens).contiguous() # get embedding for each token and reshape (..., emb_dim)

        # # Reshape to original
        z = z.reshape(*shape[:-1], *z.shape[1:]) # (..., emb_dim) 
        z_q = z_q.reshape(*shape[:-1], *z_q.shape[1:]) # (..., emb_dim)
        tokens = tokens.reshape(*shape[:-1], -1) # (...)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., emb_dim)
        z_q = z_q.view(-1, *shape[-3:]) # (..., emb_dim)
        rec = self.decoder(z_q) # (..., C)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:]) # (..., C)
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec
