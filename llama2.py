import math
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn


@dataclass
class Llama2Config:
    # default for 7b
    block_size: int = 4096  # max sequence length
    vocab_size: int = 32000  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 32  # number of layers
    n_head: int = 32  # number of heads
    n_embd: int = 4096  # embedding dimension
    norm_eps: float = 1e-5  # hyperparamter for RMSNorm
    n_hidden: int = 11008  # hidden layer dimension for FFN
    dtype: torch.dtype = torch.bfloat16  # model weights dtype


class LlamaTokenizer:
    def __init__(self, filepath: str | Path):
        sp = spm.SentencePieceProcessor()
        sp.load(filepath)
        self.tokenizer = sp

    def encode(self, text: str):
        return self.tokenizer.encode_as_ids(text)  # encode text to list of tokens

    def decode(self, ids: list):
        return self.tokenizer.decode_pieces(ids)  # decode tokens to text

    def text_to_token_ids(self, text: str):
        encoded = self.encode(text)  # encode the text into list of tokens
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids: torch.Tensor):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return self.decode(flat.tolist())  # convert tensor to string and decode


class RMSNorm(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.embd_dim = config.n_embd
        self.eps = config.norm_eps  # (1)
        self.weight = nn.Parameter(torch.ones(self.embd_dim)).float()  # (C)

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        x_normed = x * torch.rsqrt(
            means + self.eps
        )  # (B, T, C) / root((B, T, 1) + (1)) -> (B, T, C)
        return (x_normed * self.weight).to(
            dtype=x.dtype
        )  # (B, T, C) * (C) -> (B, T, C)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MHA(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False, dtype=config.dtype
        )
        self.wk = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False, dtype=config.dtype
        )
        self.wv = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False, dtype=config.dtype
        )
        self.wo = nn.Linear(
            self.config.n_embd, self.config.n_embd, bias=False, dtype=config.dtype
        )
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(1, 1, self.config.block_size, self.config.block_size)
            ),
        )
        sin, cos = precompute_rope_params(config=config)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head

        # calculate q,k,v
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        # rotate q and k
        q = apply_rope(q, sin=self.sin, cos=self.cos)
        k = apply_rope(k, sin=self.sin, cos=self.cos)

        # attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= 0, is_causal=True)
        attn_scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        attn_scores = torch.masked_fill(
            attn_scores, self.mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_out = attn_scores @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.wo(attn_out)
        return attn_out


class FFN(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.w1 = nn.Linear(
            config.n_embd, config.n_hidden, bias=False, dtype=config.dtype
        )
        self.w3 = nn.Linear(
            config.n_embd, config.n_hidden, bias=False, dtype=config.dtype
        )
        self.w2 = nn.Linear(
            config.n_hidden, config.n_embd, bias=False, dtype=config.dtype
        )
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        x1 = self.silu(self.w1(x))
        x2 = self.w3(x)
        x = self.w2(x1 * x2)
        return x


class Transformer(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.attention = MHA(config=config)
        self.feed_forward = FFN(config=config)
        self.attention_norm = RMSNorm(config=config)
        self.ffn_norm = RMSNorm(config=config)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama2(nn.Module):
    def __init__(self, config: Llama2Config | None = None):
        super().__init__()
        if not config:
            pass
        else:
            self.config = config
            self.tok_embeddings = nn.Embedding(
                config.vocab_size, config.n_embd, dtype=config.dtype
            )
            self.output = nn.Linear(
                config.n_embd, config.vocab_size, dtype=config.dtype, bias=False
            )
            self.norm = RMSNorm(config=config)
            self.layers = nn.ModuleList(
                [Transformer(config=config) for _ in range(config.n_layer)]
            )

    def forward(self, x: torch.Tensor):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor | str,
        max_new_tokens: int = 100,
        top_k: int = 1,
        temperature: float = 0.0,
        eos_id: int | None = None,
        tokenizer: LlamaTokenizer | None = None,
        device: torch.device | None = None,
    ):
        self.eval()
        if isinstance(x, str):
            if device is None or tokenizer is None:
                raise Exception(
                    "Please pass a tokenizer and device when generating from text"
                )
            x = tokenizer.text_to_token_ids(x).to(device)

        max_len = self.config.block_size
        inp_len = x.shape[-1]
        if inp_len > max_len:
            raise Exception(
                f"input length {inp_len} is greater than model's max_seq_len {max_len}"
            )

        for i in range(max_new_tokens):
            idx_cond = x[
                :, -max_len:
            ]  # important to stay in model's context at every time step
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            if temperature > 0.0:
                logits = logits / temperature  # (1, T)
                probs = torch.softmax(logits, dim=-1)  # (1, T)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:
                break

            x = torch.cat((x, idx_next), dim=1)

        text = tokenizer.token_ids_to_text(x)
        return text

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        token: str | None = None,
        save_dir_path: str | Path | None = None,
        device=torch.device,
    ):
        from huggingface_hub import hf_hub_download, login

        assert model_name in ["meta-llama/Llama-2-7b-chat"]

        if not token:
            raise Exception("Please pass your hf token to download model and tokenizer")
        else:
            login(token=token)

        if not save_dir_path:
            save_dir_path = Path.cwd()
        else:
            if isinstance(save_dir_path, str):
                save_dir_path = Path(save_dir_path)
            save_dir_path.mkdir(exist_ok=True, parents=True)

        if model_name == "meta-llama/Llama-2-7b-chat":
            tokenizer_file = hf_hub_download(
                repo_id=model_name, filename="tokenizer.model", local_dir=save_dir_path
            )

            weights_file = hf_hub_download(
                repo_id=model_name,
                filename="consolidated.00.pth",
                local_dir=save_dir_path,
            )

            config = Llama2Config()
            model = Llama2(config=config)
            model.to(device)
            weights = torch.load(
                save_dir_path / "consolidated.00.pth",
                map_location="cpu",
                weights_only=True,
            )
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param_shape = param.shape
                    weight = weights.get(name, None)
                    if weight is not None:
                        weights_shape = weight.shape
                        if param_shape == weights_shape:
                            param.copy_(weight.to(device))
                        elif param_shape == weight.T.shape:
                            param.copy_(weight.T.to(device))
                        else:
                            print(
                                f"Shape of weight {weights_shape} mismatch with model weight {param_shape} for {name}"
                            )
                    else:
                        pass

        tokenizer = LlamaTokenizer(filepath=str(save_dir_path / "tokenizer.model"))
        return model, tokenizer


def precompute_rope_params(config):
    T, C, n = config.block_size, config.n_embd, config.n_head
    h = C // n
    positions = torch.arange(0, T)  # (T, 1)
    thetas = 1.0 / (10000 ** (torch.arange(0, h, 2, dtype=torch.float32) / h))
    pos_thetas = torch.outer(positions, thetas)  # (T, h/2)
    pos_thetas = torch.cat([pos_thetas, pos_thetas], dim=-1)  # (T,h)
    cos = pos_thetas.cos()  # (T, h)
    sin = pos_thetas.sin()  # (T, h)
    return sin, cos


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    B, n, T, h = x.shape
    rotated_x = (
        x * cos[:T, :]
        + torch.cat([-x[..., h // 2 :], x[..., : h // 2]], dim=-1) * sin[:T, :]
    )
    return rotated_x.to(dtype=x.dtype, device=x.device)
