import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import PositionalEncoding, TransformerBlock

class SLM(nn.Module):
    """
    Small Language Model (SLM) with optional pretrained embeddings,
    learned vs. sinusoidal positional encoding, and sliding-window attention.
    """
    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        max_seq_len: int,
        dropout_p: float,
        n_heads: int,
        n_blocks: int,
        context_len: int,
        vocab_size: int,
        pretrained_embeddings: torch.Tensor = None,
        sliding_window: int = None,
        learned_pe: bool = False,
        freeze_embeddings: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the SLM model.

        Args:
            d_model (int): Dimension of the model.
            embed_dim (int): Dimension of the embeddings.
            max_seq_len (int): Maximum sequence length.
            dropout_p (float): Dropout probability.
            n_heads (int): Number of attention heads.
            n_blocks (int): Number of transformer blocks.
            context_len (int): Context length for generation.
            vocab_size (int): Size of the vocabulary.
            pretrained_embeddings (torch.Tensor, optional): Pretrained embedding weights.
            sliding_window (int, optional): Size of sliding window for attention.
            learned_pe (bool): Use learned positional encoding if True, else sinusoidal.
            freeze_embeddings (bool): Freeze embedding weights if True.
            verbose (bool): Print initialization messages if True.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("Initializing SLM model")

        self.d_model = d_model
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.context_len = context_len
        self.n_blocks = n_blocks
        self.learned_pe = learned_pe
        self.pretrained_embeddings = pretrained_embeddings
        self.sliding_window = sliding_window
        self.t = None 

        self.token_embedding = self._build_token_embedding(vocab_size, embed_dim, freeze_embeddings)

        self.pos_encoding = (
            nn.Embedding(max_seq_len, embed_dim)
            if learned_pe
            else PositionalEncoding(d_model=embed_dim, max_seq_len=max_seq_len, dropout_p=dropout_p)
        )

        self.transformers = nn.Sequential(*[
            TransformerBlock(
                d_model=d_model,
                embed_dim=embed_dim,
                n_heads=n_heads,
                dropout_p=dropout_p,
                sliding_window=sliding_window if i % 2 == 1 else None  
            )
            for i in range(n_blocks)
        ])

        self.output_linear = nn.Linear(d_model, vocab_size, bias=True)
        self.output_linear.weight = self.token_embedding.weight

        self._init_weights()

    def _build_token_embedding(self, vocab_size, embed_dim, freeze):
        """
        Build token embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embeddings.
            freeze (bool): Freeze embedding weights if True.

        Returns:
            nn.Embedding: Token embedding layer.
        """
        if isinstance(self.pretrained_embeddings, torch.Tensor):
            if self.verbose:
                print("Loading pretrained embeddings")
            return nn.Embedding.from_pretrained(self.pretrained_embeddings, freeze=freeze)
        else:
            return nn.Embedding(vocab_size, embed_dim)

    def _init_weights(self):
        """
        Initialize model weights using Xavier Normal for Linear layers and normal
        initialization for Embedding layers if no pretrained embeddings are provided.
        """
        if self.verbose:
            print("Applying Xavier Normal init to Linear, normal init to Embedding")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and self.pretrained_embeddings is None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids (torch.LongTensor): Input token IDs.

        Returns:
            torch.Tensor: Token embeddings.
        """
        return self.token_embedding(token_ids)

    def add_positional_encoding(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            t (int, optional): Time index for positional encoding. If None, use full range.

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        if not self.learned_pe:
            return self.pos_encoding(x, t)
        seq_len = x.size(1)
        if t is None:
            positions = torch.arange(seq_len, device=x.device)
        else:
            positions = torch.tensor([t], device=x.device)
        pe = self.pos_encoding(positions).unsqueeze(0)
        return x + pe

    def apply_transformers(self, x: torch.Tensor, inference: bool, first: bool) -> torch.Tensor:
        """
        Apply transformer blocks sequentially.

        Args:
            x (torch.Tensor): Input tensor.
            inference (bool): Whether in inference mode.
            first (bool): Whether this is the first step in inference.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        for layer in self.transformers:
            x = layer(x, _inference=inference, _first=first)
            first = False  
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits over vocabulary.
        """
        return self.output_linear(x)

    def forward(self, tokens: torch.LongTensor, _inference: bool = False, _first: bool = False):
        """
        Perform forward pass through the model.

        Args:
            tokens (torch.LongTensor): Input token IDs.
            _inference (bool): Whether in inference mode.
            _first (bool): Whether this is the first step in inference.

        Returns:
            torch.Tensor: Output logits.
        """
        if _inference and not _first:
            tokens = tokens[:, -1:]

        x = self.embed(tokens)

        if _inference:
            x = self.add_positional_encoding(x, self.t)
            if _first:
                self.t = x.size(1)
            else:
                self.t += 1
        else:
            x = self.add_positional_encoding(x, None)

        x = self.apply_transformers(x, inference=_inference, first=_first)

        return self.project(x)

    def generate(
        self,
        X: torch.LongTensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_toks: int,
        context_len: int,
        eos_token: int,
        greedy: bool = False
    ):
        """
        Generate token IDs autoregressively.

        Args:
            X (torch.LongTensor): Input token IDs (batch size = 1).
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling threshold.
            top_k (int): Top-k sampling threshold.
            max_toks (int): Maximum number of tokens to generate.
            context_len (int): Context length for generation.
            eos_token (int): End-of-sequence token ID.
            greedy (bool): Use greedy sampling if True.

        Yields:
            int: Generated token ID.
        """
        assert X.size(0) == 1, "Only single-batch generation supported"
        _first, _inference = True, True

        if X.size(1) > context_len:
            X = X[:, -context_len:]

        for _ in range(max_toks):
            logits = self.forward(X, _inference=_inference, _first=_first)
            next_tok = self.sample_model(logits, temperature, top_p, top_k, greedy)
            if next_tok.item() == eos_token:
                break
            yield next_tok.item()

            X = torch.cat([X, next_tok], dim=1)
            if X.size(1) > context_len:
                X = X[:, -context_len:]
            _first = False

    def sample_model(
        self,
        logits: torch.Tensor,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        greedy: bool = False
    ) -> torch.Tensor:
        """
        Sample the next token from logits.

        Args:
            logits (torch.Tensor): Input logits.
            temperature (float, optional): Sampling temperature.
            top_p (float, optional): Top-p sampling threshold.
            top_k (int, optional): Top-k sampling threshold.
            greedy (bool): Use greedy sampling if True.

        Returns:
            torch.Tensor: Sampled token ID.
        """
        logits = logits[:, -1, :]

        if temperature and temperature != 0:
            logits = logits / temperature

        if top_k is not None:
            if top_k >= logits.size(-1):
                raise ValueError("top_k must be < vocab size")
            values, indices = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, indices, values)

        if top_p is not None:
            if not 0 <= top_p <= 1:
                raise ValueError("top_p must be in [0, 1]")
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            cutoff = cum_probs <= top_p
            cutoff[:, 0] = True  # always keep highest
            mask = cutoff.scatter(-1, sorted_indices.argsort(dim=-1), cutoff)
            logits = logits.masked_fill(~mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        if greedy:
            return torch.argmax(probs, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)