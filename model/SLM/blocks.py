import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Implements positional encoding for transformer models."""
    def __init__(self, d_model: int, max_seq_len: int, dropout_p: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model (int): Model dimension.
            max_seq_len (int): Maximum sequence length.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout_p)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = 10000 ** (i / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, t=None):
        """Apply positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            t (int, optional): Specific time step. Defaults to None.

        Returns:
            torch.Tensor: Input tensor with positional encoding applied.

        Raises:
            ValueError: If sequence length exceeds max_seq_len or invalid seq_len when t is specified.
        """
        if t is None:
            if x.size(1) > self.max_seq_len:
                raise ValueError(f"Sequence length {x.size(1)} exceeds maximum {self.max_seq_len}")
            return self.dropout(x + self.pe[:, :x.size(1), :].to(x.device))
        else:
            if t >= self.max_seq_len:
                t = self.max_seq_len - 1
            if x.size(1) != 1:
                raise ValueError(f"When t is specified, seq_len must be 1, got {x.size(1)}")
            return self.dropout(x + self.pe[:, t:t+1, :].to(x.device))


class TransformerBlock(nn.Module):
    """Implements a single transformer block with multi-head self-attention."""
    def __init__(self, d_model: int, embed_dim: int, n_heads: int, dropout_p: float, sliding_window: int = None):
        """Initialize transformer block.

        Args:
            d_model (int): Model dimension.
            embed_dim (int): Embedding dimension.
            n_heads (int): Number of attention heads.
            dropout_p (float): Dropout probability.
            sliding_window (int, optional): Sliding window size for attention. Defaults to None.
        """
        super().__init__()
        self.n_heads = n_heads
        self.sliding_window = sliding_window

        self.linearQ = nn.Linear(embed_dim, d_model)
        self.linearK = nn.Linear(embed_dim, d_model)
        self.linearV = nn.Linear(embed_dim, d_model)

        self.MHSA = MultiHeadSelfAttention(d_model=d_model, dropout_p=dropout_p, sliding_window=sliding_window)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.pffn = PositionWiseFNN(d_model=d_model, dropout_p=dropout_p)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, _inference=False, _first=False):
        """Process input through transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            _inference (bool, optional): Whether in inference mode. Defaults to False.
            _first (bool, optional): Whether first token in sequence. Defaults to False.

        Returns:
            torch.Tensor: Processed tensor.
        """
        if _inference and not _first:
            if x.shape[1] != 1:
                x = x[:, -1:, :]
                x = self.layernorm1(x)
            q = self.linearQ(x)
            k = self.linearK(x)
            v = self.linearV(x)
        else:
            self.__c_seq_len = x.shape[1]
            x = self.layernorm1(x)
            q = self.linearQ(x)
            k = self.linearK(x)
            v = self.linearV(x)

        batch_size, seq_len, _ = q.shape
        head_dim = q.shape[-1] // self.n_heads

        q = q.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        context_len = self.sliding_window if self.sliding_window else self.__c_seq_len
        x_attn = self.MHSA(q, k, v, _inference=_inference, context_len=context_len, _first=_first)

        x = x_attn + x
        x_res = x
        x = self.layernorm2(x)
        x = self.pffn(x) + x_res

        return x


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention mechanism."""
    def __init__(self, dropout_p: float, d_model: int, sliding_window: int = None):
        """Initialize multi-head self-attention.

        Args:
            dropout_p (float): Dropout probability.
            d_model (int): Model dimension.
            sliding_window (int, optional): Sliding window size for attention. Defaults to None.
        """
        super().__init__()
        self.sliding_window = sliding_window
        self.dropout = nn.Dropout(p=dropout_p)
        self.linearO = nn.Linear(d_model, d_model)

    def create_swa_mask(self, mask, seq_len, k_seq_len=None, square=True):
        """Create sliding window attention mask.

        Args:
            mask (torch.Tensor): Initial mask tensor.
            seq_len (int): Sequence length.
            k_seq_len (int, optional): Key sequence length. Defaults to None.
            square (bool, optional): Whether to create square mask. Defaults to True.

        Returns:
            torch.Tensor: Modified mask tensor.
        """
        if square:
            for i in range(seq_len):
                if i >= self.sliding_window:
                    mask[i, :i - self.sliding_window] = 1
        else:
            for i in range(seq_len):
                for j in range(k_seq_len):
                    if j - self.sliding_window > 0:
                        mask[i, :j - self.sliding_window] = 1
        return mask

    def forward(self, q, k, v, context_len: int = None, _inference=False, _first: bool = None):
        """Apply multi-head self-attention.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            context_len (int, optional): Context length. Defaults to None.
            _inference (bool, optional): Whether in inference mode. Defaults to False.
            _first (bool, optional): Whether first token in sequence. Defaults to None.

        Returns:
            torch.Tensor: Attention output.

        Raises:
            RuntimeWarning: If seq_len != 1 during inference.
            ValueError: If K_cache and V_cache shapes mismatch.
        """
        device = q.device
        seq_len = q.shape[2]

        if not _inference:
            mask = torch.ones(seq_len, seq_len, device=device)
            mask = torch.triu(mask, diagonal=1)
            if self.sliding_window:
                mask = self.create_swa_mask(mask, seq_len)

            attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
            attn_scores = self.dropout(F.softmax(attn_logits, dim=-1))
            attn_output = torch.matmul(attn_scores, v)

        else:
            q_seq_len = seq_len
            k_seq_len = k.shape[2] if _first else self.K_cache.shape[2] + k.shape[2]

            if _first:
                self.K_cache = k
                self.V_cache = v
                mask = torch.ones(q_seq_len, k_seq_len, device=device)
                mask = torch.triu(mask, diagonal=1)
                if self.sliding_window:
                    mask = self.create_swa_mask(mask, q_seq_len, k_seq_len, square=True)
            else:
                if q_seq_len != 1:
                    raise RuntimeWarning('seq_len != 1 during inference.')
                if self.K_cache.shape != self.V_cache.shape:
                    raise ValueError('K_cache and V_cache shapes mismatch.')

                if self.K_cache.shape[2] >= context_len:
                    self.K_cache = self.K_cache[:, :, -(context_len - 1):, :]
                    self.V_cache = self.V_cache[:, :, -(context_len - 1):, :]

                self.K_cache = torch.cat([self.K_cache, k], dim=2)
                self.V_cache = torch.cat([self.V_cache, v], dim=2)
                k_seq_len = self.K_cache.shape[2]
                mask = torch.zeros(q_seq_len, k_seq_len, device=device)
                if self.sliding_window:
                    mask[:, :max(0, k_seq_len - self.sliding_window)] = 1

            k = self.K_cache
            v = self.V_cache

            self.K_cache = self.K_cache.to('cpu')
            self.V_cache = self.V_cache.to('cpu')

            attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
            attn_scores = self.dropout(F.softmax(attn_logits, dim=-1))
            attn_output = torch.matmul(attn_scores, v)

        x = self.dropout(self.linearO(attn_output.transpose(1, 2).contiguous().view(
            attn_output.shape[0], attn_output.shape[2], -1)))
        return x


class PositionWiseFNN(nn.Module):
    """Implements position-wise feed-forward network."""
    def __init__(self, d_model: int, dropout_p: float):
        """Initialize position-wise feed-forward network.

        Args:
            d_model (int): Model dimension.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """Apply position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        x = F.gelu(self.linear1(x))
        return self.dropout(self.linear2(x))