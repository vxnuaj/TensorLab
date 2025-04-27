import torch
import torch.nn as nn
import warnings

from model.blocks import PositionalEmbedding, TransformerBlock

class LLaMA(nn.Module):
    """
    THe LlaMA architecture, supporting various attention types and positional embeddings.

    Attributes:
        context_len (int): Maximum sequence length the model can process.
        d_model (int): Dimensionality of the input and output features.
        n_heads (int): Number of attention heads in each transformer block.
        n_blocks (int): Number of transformer blocks.
        vocab_size (int): Size of the vocabulary for token embeddings.
        pos_emb_dropout_p (float): Dropout probability for positional embeddings.
        pos_emb_type (str): Type of positional embedding ('rope' or 'pe').
        learned (bool): Whether to use learned positional embeddings (only applies if pos_emb_type is 'pe').
        ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling, if used.
        dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE, if used.
        attn_type (str): Type of attention mechanism ('mhsa', 'mqa', or 'gqa').
        n_groups (int): Number of groups for grouped query attention, if applicable.
        embeddings (nn.Embedding): Token embedding layer.
        pe (PositionalEmbedding, optional): Positional embedding layer (if pos_emb_type is 'pe').
        block (nn.ModuleList): List of transformer blocks.
        rmsnorm (nn.RMSNorm): RMS normalization layer.
        linear (nn.Linear): Final linear layer for vocabulary prediction.
    """

    def __init__(
        self,
        context_len: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        vocab_size: int,
        pos_emb_dropout_p: float = 0.1,
        pos_emb_type: str = "rope",
        learned: bool = False,
        ntk_rope_scaling: bool = False,
        dyn_scaling: bool = False,
        attn_type: str = "gqa",
        n_groups: int = None,
        top_k_sparsev:int = None,
        p_threshold:int = None,
        p_threshold_steps_fraction:float = None,
        flash_attn:bool = False,
        flash_attn_dtype:torch.dtype = torch.float16,
        supress_warnings: bool = True,
        verbose:bool = False,
        *args,
        **kwargs
        ):
        
        """
        Initializes the LLaMA model.

        Args:
            context_len (int): Maximum sequence length.
            d_model (int): Dimensionality of the input and output features.
            n_heads (int): Number of attention heads per block.
            n_blocks (int): Number of transformer blocks.
            vocab_size (int): Size of the vocabulary.
            pos_emb_dropout_p (float, optional): Dropout probability for positional embeddings. Defaults to 0.1.
            pos_emb_type (str, optional): Positional embedding type ('rope' or 'pe'). Defaults to 'rope'.
            learned (bool, optional): If True, uses learned positional embeddings (only for 'pe'). Defaults to False.
            ntk_rope_scaling (Union[dict, bool], optional): If dict, contains 'pretrained_context_window' and
                'new_context_window' for NTK RoPE scaling; if False, no scaling. Defaults to False.
            dyn_scaling (Union[bool, float], optional): If float between 0 and 1, applies dynamic RoPE scaling; if False, no scaling. Defaults to False.
            attn_type (str, optional): Attention mechanism type ('mhsa', 'mqa', 'gqa'). Defaults to 'gqa'.
            n_groups (int, optional): Number of groups for grouped query attention (required if attn_type is 'gqa'). Defaults to None.
            supress_warnings (bool, optional): If True, suppresses warnings (e.g., for 'rope' with learned). Defaults to True.
            flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
            flash_attn_dtype (torch.dtype, optional): Data type for FlashAttention. Defaults to torch.float16.

        Raises:
            ValueError: If pos_emb_type is not 'rope' or 'pe'.
            AssertionError: If pos_emb_type is not a string.
        """
        super().__init__()
       
        self._supress_warnings(supress_warnings) 
        self._check_pos_emb_type(pos_emb_type)

        self.context_len = context_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.pos_emb_dropout_p = pos_emb_dropout_p
        self.pos_emb_type = pos_emb_type
        self.learned = learned
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.attn_type = attn_type
        self.n_groups = n_groups
        self.top_k_sparsev = top_k_sparsev
        self.p_threshold = p_threshold
        self.p_threshold_steps_fraction = p_threshold_steps_fraction
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
        self.verbose = verbose

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )

        if self.pos_emb_type == 'pe':
            self.pe = PositionalEmbedding(
                context_len=self.context_len,
                d_model=self.d_model,
                dropout_p=self.pos_emb_dropout_p,
                learned=self.learned
            ) 

        self.block = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                context_len=self.context_len,
                ntk_rope_scaling=self.ntk_rope_scaling,
                dyn_scaling=self.dyn_scaling,
                attn_type=self.attn_type,
                n_groups=self.n_groups,
                top_k_sparsev = self.top_k_sparsev,
                p_threshold = self.p_threshold,
                p_threshold_steps_fraction = self.p_threshold_steps_fraction,
                flash_attn = self.flash_attn,
                flash_attn_dtype = self.flash_attn_dtype
            )
            for _ in range(self.n_blocks)
        ])
        
        self.rmsnorm = nn.RMSNorm(normalized_shape=self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        self.linear.weight = self.embeddings.weight
 
        self._init_weights()
  
    def forward(self, x, _inference: bool = False):
        """
        Processes the input through the SmolLLaMA model to generate logits over the vocabulary.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices.
            _inference (bool, optional): If True, enables inference mode with caching. Defaults to False.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embeddings(x)
      
        if self.pos_emb_type == 'pe':
            x = self.pe(x, _inference=_inference)

        for i, _ in enumerate(self.block):
            x = self.block[i](x, _inference=_inference)
            
        x = self.rmsnorm(x)
        x = self.linear(x) 
         
        return x
    
    def _init_weights(self):
        if self.verbose:
            print(f"Initializing weights using Xavier Uniform Init.")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def _check_pos_emb_type(self, pos_emb_type: str):
        """
        Validates the positional embedding type and issues a warning if 'rope' is used with learned embeddings.

        Args:
            pos_emb_type (str): Type of positional embedding ('rope' or 'pe').

        Raises:
            AssertionError: If pos_emb_type is not a string.
            ValueError: If pos_emb_type is not 'rope' or 'pe'.
        """
        assert isinstance(pos_emb_type, str), "pos_emb_type should be a string"
        if pos_emb_type not in ["rope", "pe"]:
            raise ValueError('pos_emb_type should be either "rope" or "pe"')
        if pos_emb_type == 'rope':
            warnings.warn("Using rotary positional embedding, learned will have no effect")            

    def _supress_warnings(self, supress_warnings: bool):
        """
        Configures warning suppression.

        Args:
            supress_warnings (bool): If True, suppresses all warnings; if False, enables default warning behavior.
        """
        
        assert isinstance(supress_warnings, bool), ValueError("supress_warnings must be type bool")
        
        if supress_warnings:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")