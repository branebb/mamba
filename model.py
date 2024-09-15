from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from transformers import AutoTokenizer

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        # Calculate inner dimension
        self.d_inner = int(self.expand * self.d_model)

        # Set dt_rank if it's set to 'auto'
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.args = args
        self.device = device

        # Embedding layer
        self.embedding = nn.Embedding(args.vocab_size, args.d_model).to(device)
        
        # Create residual blocks
        self.layers = nn.ModuleList([ResidualBlock(args, device=device) for _ in range(args.n_layer)]).to(device)
        
        # RMS normalization
        self.norm_f = RMSNorm(args.d_model, device=device).to(device)
        
        # Linear layer for language modeling head
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False).to(device)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids.to(self.device))
        
        # Pass through each residual block
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str, device: torch.device = torch.device('cpu')):
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        # Load configuration and state dict
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args, device=device)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {key.replace('backbone.', ''): value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)

        return model

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.args = args
        
        # Initialize MambaBlock and RMSNorm
        self.mixer = MambaBlock(args, device=device).to(device)
        self.norm = RMSNorm(args.d_model, device=device).to(device)

    def forward(self, x):
        # Apply normalization and MambaBlock with residual connection
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.args = args
        self.device = device

        # Linear projection for input
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias).to(device)

        # 1D Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        ).to(device)

        # Linear projections for state space model
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False).to(device)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True).to(device)

        # Parameters for state space model
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner).to(device)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner).to(device))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias).to(device)

    def forward(self, x):
        (b, l, d) = x.shape

        # Split input projection into x and residual components
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        # Apply convolution and activation function
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        # Apply state space model
        y = self.ssm(x)

        # Apply gating and output projection
        y = y * F.silu(res)
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        # Convert log of A to A and get D
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # Apply linear projection
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # Apply selective scan
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Compute deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform the selective scan
        x = torch.zeros((b, d_in, n), device=self.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)

        # Add residual component
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model).to(device))

    def forward(self, x):
        # Apply RMS normalization
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40,
             device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):  # Added device
    model.eval()

    # Move input_ids to the correct device
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        # Ensure next_indices is on the same device
        input_ids = torch.cat([input_ids, next_indices], dim=1).to(device)

    # Decode the output to text, make sure the result is on CPU
    output_completions = tokenizer.decode(input_ids[0].tolist())

    return output_completions