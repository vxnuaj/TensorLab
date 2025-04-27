import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Union

class TransformerBlock(nn.Module):
    """
    A single block of a Transformer model, consisting of an attention mechanism followed by a feed-forward network,
    both with residual connections and RMS normalization.

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        n_heads (int): Number of attention heads.
        d_head (int): Dimensionality of each attention head (d_model // n_heads).
        context_len (int): Maximum sequence length the model can handle.
        ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling, if used.
        dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE, if used.
        attn_type (str): Type of attention mechanism ('mhsa', 'mqa', or 'gqa').
        n_groups (int): Number of groups for grouped query attention, if applicable.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        context_len: int,
        ntk_rope_scaling: Union[dict, bool], 
        dyn_scaling: Union[bool, float], 
        attn_type: str = "gqa",
        n_groups: int = None,
        top_k_sparsev:int = None,
        p_threshold:int = None,
        p_threshold_steps_fraction:float = None,
        flash_attn: bool = False,
        flash_attn_dtype:torch.dtype = torch.float16
        ):
        
        """
        Initializes the TransformerBlock.

        Args:
            d_model (int): Dimensionality of the input and output features.
            n_heads (int): Number of attention heads.
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool]): If dict, contains 'pretrained_context_window' and 'new_context_window'
                for NTK RoPE scaling; if False, no scaling is applied.
            dyn_scaling (Union[bool, float]): If float between 0 and 1, applies dynamic scaling to RoPE; if False, no scaling.
            attn_type (str, optional): Attention mechanism type ('mhsa', 'mqa', 'gqa', 'mva'). Defaults to 'gqa'.
            n_groups (int, optional): Number of groups for grouped query attention. Required if attn_type is 'gqa'.
            flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
            flash_attn_dtype (torch.dtype, optional): Data type for FlashAttention. Defaults to torch.float16.
        """
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads 
        self.d_head = d_model // n_heads
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling 
        self.attn_type = attn_type
        self.n_groups = n_groups
        self.top_k_sparsev = top_k_sparsev
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
        self.p_threshold = p_threshold
        self.p_threshold_steps_fraction = p_threshold_steps_fraction
      
        self.rmsnorm1 = nn.RMSNorm(normalized_shape=d_model) 
        self.linearQ = nn.Linear(d_model, d_model)
        self.linearK, self.linearV = self._get_attn_projs()
        self.attention = self._get_attention()
        self.rmsnorm2 = nn.RMSNorm(normalized_shape=d_model)
        self.swigluNN = FeedForwardSwiGLU(d_model=d_model, h_dim = int((2/3) * 4 * d_model))
        
    def forward(self, x, _inference=False):
        """
        Processes the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            _inference (bool, optional): If True, uses caching for keys and values during inference. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        x_res = x 
        x = self.rmsnorm1(x)
        
        q = self.linearQ(x)  
        k = self.linearK(x)
        v = self.linearV(x)
       
        if _inference and (not hasattr(self, 'k_cache') or self.k_cache is None):
            self.k_cache = k.to(torch.float16)
            self.v_cache = v.to(torch.float16)
        elif _inference and (hasattr(self, 'k_cache') or self.k_cache is not None):
            assert q.shape[1] == 1, f"Expected q sequence length of 1 once KV cache exists, got {q.shape[1]}" 
            assert k.shape[1] == 1, f"Expected k sequence length of 1 once KV cache exists, got {k.shape[1]}"
            assert v.shape[1] == 1, f"Expected v sequence length of 1 once KV cache exists, got {v.shape[1]}"
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1) 
            if self.attn_type not in ['mva', 'tksva']:
                assert self.k_cache.shape == self.v_cache.shape, f"Expected k_cache and v_cache to match, got {self.k_cache.shape}, {self.v_cache.shape}" 
            if self.k_cache.shape[1] > self.context_len:
                self.k_cache = self.k_cache[:, -self.context_len:, :]
                self.v_cache = self.v_cache[:, -self.context_len:, :] 
            k = self.k_cache
            v = self.v_cache

        x = self.attention(q, k, v, _inference=_inference) + x_res
        
        x = self.rmsnorm2(x) 
        x = self.swigluNN(x) + x 
        return x
    
    def _get_attention(self):
        """
        Returns the appropriate attention mechanism based on attn_type.

        Returns:
            nn.Module: The attention mechanism instance.
        
        Raises:
            AssertionError: If attn_type is not 'mhsa', 'mqa', or 'gqa'.
            ValueError: If n_groups is None when attn_type is 'gqa'.
        """
        assert self.attn_type in ['mhsa', 'mqa', 'gqa', 'mva', 'tksva'], f"Invalid attention type: {self.attn_type}. Choose from 'mhsa', 'mqa', 'gqa', 'mva', 'tksva'"

        if self.attn_type == 'mhsa':
            return MultiHeadSelfAttention(
                n_heads=self.n_heads,
                d_model=self.d_model,
                context_len=self.context_len,
                ntk_rope_scaling=self.ntk_rope_scaling,
                dyn_scaling=self.dyn_scaling,
                flash_attn = self.flash_attn,
                flash_attn_dtype = self.flash_attn_dtype
            )  

        elif self.attn_type == 'mqa':
            return MultiQueryAttention(
                n_heads=self.n_heads,
                d_model=self.d_model,
                context_len=self.context_len,
                ntk_rope_scaling=self.ntk_rope_scaling,
                dyn_scaling=self.dyn_scaling,
                flash_attn = self.flash_attn,
                flash_attn_dtype = self.flash_attn_dtype
            )

        elif self.attn_type == 'gqa':
            if self.n_groups is None:
                raise ValueError("n_groups must be specified for Grouped Query Attention") 
            return GroupedQueryAttention(
                n_heads=self.n_heads,
                n_groups=self.n_groups,
                d_model=self.d_model,
                context_len=self.context_len,
                ntk_rope_scaling=self.ntk_rope_scaling,
                dyn_scaling=self.dyn_scaling,
                flash_attn = self.flash_attn,
                flash_attn_dtype = self.flash_attn_dtype
            )

        elif self.attn_type == 'mva':
            return MultiValueAttention(
                n_heads = self.n_heads,
                d_model = self.d_model,
                context_len = self.context_len,
                ntk_rope_scaling = self.ntk_rope_scaling,
                dyn_scaling = self.dyn_scaling,
            )

        elif self.attn_type == 'tksva':
            return TopKSparseVAttention(
                n_heads = self.n_heads,
                d_model = self.d_model,
                top_k_sparsev = self.top_k_sparsev,
                context_len = self.context_len,
                ntk_rope_scaling = self.ntk_rope_scaling,
                dyn_scaling = self.dyn_scaling, 
            )

    def _get_attn_projs(self):
        """
        Returns linear projections for keys and values based on attn_type.

        Returns:
            tuple: (linearK, linearV), the linear layers for keys and values.
        
        Raises:
            AssertionError: If attn_type is not 'mhsa', 'mqa', or 'gqa'.
            ValueError: If n_groups is None when attn_type is 'gqa'.
        """
        assert self.attn_type in ['mhsa', 'mqa', 'gqa', 'mva', 'tksva'], f"Invalid attention type: {self.attn_type}. Choose from ['mhsa', 'mqa', 'gqa', 'mva', 'tksva']"
        if self.attn_type == 'mhsa':
            linearK = nn.Linear(self.d_model, self.d_model)
            linearV = nn.Linear(self.d_model, self.d_model)              
            return linearK, linearV
        elif self.attn_type == 'mqa':
            linearK = nn.Linear(self.d_model, self.d_head)
            linearV = nn.Linear(self.d_model, self.d_head)              
            return linearK, linearV 
        elif self.attn_type == 'gqa':
            if self.n_groups is None:
                raise ValueError("n_groups must be specified for Grouped Query Attention") 
            assert self.d_model % self.n_groups == 0, f"Expected d_model divisible by n_groups, got d_model: {self.d_model}, n_groups: {self.n_groups}"
            linearK = nn.Linear(self.d_model, self.d_head * self.n_groups)
            linearV = nn.Linear(self.d_model, self.d_head * self.n_groups) 
            return linearK, linearV 
        elif self.attn_type == 'mva':
            linearK = nn.Linear(self.d_model, self.d_head)
            linearV = nn.Linear(self.d_model, self.d_model)
            return linearK, linearV
        elif self.attn_type == 'tksva':
            linearK = nn.Linear(self.d_model, self.d_head)
            linearV = nn.Linear(self.d_model, self.d_model)
            return linearK, linearV

    def _reset_cache(self):
        """
        Resets the key and value caches used during inference.
        """
        if hasattr(self, 'k_cache'):
            del self.k_cache
        if hasattr(self, 'v_cache'):
            del self.v_cache

class MultiHeadSelfAttention(nn.Module):
    """
    Implements standard multi-head self-attention with rotary positional embeddings.

    Attributes:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the input features.
        d_head (int): Dimensionality of each attention head (d_model // n_heads).
        context_len (int): Maximum sequence length.
        ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling.
        dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE.
    """
    
    def __init__(
        self, 
        n_heads: int, 
        d_model: int, 
        context_len: int, 
        ntk_rope_scaling: Union[dict, bool], 
        dyn_scaling: Union[bool, float],
        flash_attn:bool = False,
        flash_attn_dtype:torch.dtype = torch.float16
        ):
        """
        Initializes the MultiHeadSelfAttention module.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the input features.
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool]): If dict, contains 'pretrained_context_window' and 'new_context_window'
                for NTK RoPE scaling; if False, no scaling.
            dyn_scaling (Union[bool, float]): If float between 0 and 1, applies dynamic scaling to RoPE; if False, no scaling.
            flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
            flash_attn_dtype (torch.dtype, optional): Data type for FlashAttention. Defaults to torch.float16.            
        """

        super().__init__()
        self.n_heads = n_heads 
        self.d_model = d_model
        self.d_head = d_model // n_heads 
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
        self.rope = RotaryPositionalEmbedding(
            d_head=self.d_head,
            context_len=self.context_len,
            ntk_rope_scaling=self.ntk_rope_scaling,
            dyn_scaling=self.dyn_scaling
        )
        
    def forward(self, q, k, v, _inference=False):
        """
        Computes multi-head self-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, d_model).
            _inference (bool, optional): If True, adjusts RoPE for inference mode. Defaults to False.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, sequence_length, d_model).
        """
   
        b, q_l, d_model = q.shape
        _, k_l, _ = k.shape 
        _, v_l, _ = v.shape 
 
        if _inference:
            seq_len = q_l
        else:
            seq_len = self.context_len
        
        assert d_model == self.d_model, f"Expected d_model to be {self.d_model}, got {d_model}" 
        
        d_head = d_model // self.n_heads
        
        q = q.view(b, self.n_heads, q_l, d_head)
        k = k.view(b, self.n_heads, k_l, d_head)
        v = v.view(b, self.n_heads, v_l, d_head)

        q = self.rope(q, _inference=_inference, _q=True)
        k = self.rope(k, _inference=_inference)

        if self.flash_attn:
            q = q.to(self.flash_attn_dtype)
            k = k.to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype) 

        attn_output = F.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            is_causal = True,
            enable_gqa = False
        ).contiguous().view(b, seq_len, d_model).to(torch.float32)

        '''
        if self.flash_attn:
            
            q = q.to(self.flash_attn_dtype)
            k = k.to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype) 
            
            attn_output = F.scaled_dot_product_attention(
                query = q,
                key = k,
                value = v,
                is_causal = True,
                enable_gqa = False
            ).view(b, l, d_model)
            
        else:
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
            causal_mask = torch.tril(torch.ones(attn_logits.shape[-1], attn_logits.shape[-1]), diagonal=0).unsqueeze(0).unsqueeze(0).bool().to(self.device)
            attn_logits = attn_logits.masked_fill(causal_mask == 0, float("-inf")) 
            attn_scores = F.softmax(attn_logits, dim=-1)
            attn_output = torch.matmul(attn_scores, v).view(b, l, d_model)
        '''
        
        return attn_output

class MultiQueryAttention(nn.Module):
    """
    Implements multi-query attention where keys and values are shared across heads.

    Attributes:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the input features.
        d_head (int): Dimensionality of each attention head (d_model // n_heads).
        context_len (int): Maximum sequence length.
        ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling.
        dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE.
        flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
    """
    
    def __init__(
        self, 
        n_heads: int, 
        d_model: int, 
        context_len: int,
        ntk_rope_scaling: Union[dict, bool], 
        dyn_scaling: Union[bool, float],
        flash_attn:bool = False,
        flash_attn_dtype:torch.dtype = torch.float16
        ):
        """
        Initializes the MultiQueryAttention module.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the input features.
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool]): If dict, contains 'pretrained_context_window' and 'new_context_window'
                for NTK RoPE scaling; if False, no scaling.
            dyn_scaling (Union[bool, float]): If float between 0 and 1, applies dynamic scaling to RoPE; if False, no scaling.
            flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
            flash_attn_dtype (torch.dtype, optional): Data type for FlashAttention. Defaults to torch.float16.
        """
        super().__init__()
        self.n_heads = n_heads 
        self.d_model = d_model
        self.d_head = d_model // n_heads 
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
        self.rope = RotaryPositionalEmbedding(
            d_head=self.d_head,
            context_len=self.context_len,
            ntk_rope_scaling=self.ntk_rope_scaling,
            dyn_scaling=self.dyn_scaling
        )
   
    def forward(self, q, k, v, _inference=False):
        """
        Computes multi-query attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, d_head).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, d_head).
            _inference (bool, optional): If True, adjusts RoPE for inference mode. Defaults to False.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, sequence_length, d_model).
        """
        b, seq_len, d_model = q.shape
        assert d_model == self.d_model, f"Expected d_model to be {self.d_model}, got {d_model}" 
        
        q = q.view(b, self.n_heads, seq_len, self.d_head)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1) 
       
        if not _inference:
            assert q.shape[2:] == k.shape[2:], f"Expected q and k to match on seq_len and d_head, got {q.shape}, {k.shape}" 
        assert k.shape == v.shape, f"Expected k and v to have the same shape, got {k.shape}, {v.shape}"
        
        q = self.rope(q, _inference=_inference, _q=True)
        k = self.rope(k, _inference=_inference)

        if self.flash_attn: 
            # flash attn only works with float16 or bfloat16
            q = q.to(self.flash_attn_dtype)
            k = k.to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype)
            
        attn_output = F.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            is_causal = True,
            enable_gqa = True
        ).contiguous().view(b, seq_len, d_model).to(torch.float32)
       
        '''
        if self.flash_attn: 
           
            # flash attn only works with float16 or bfloat16
            q = q.to(self.flash_attn_dtype)
            k = k.to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype)
            
            attn_output = F.scaled_dot_product_attention(
                query = q,
                key = k,
                value = v,
                is_causal = True,
                enable_gqa = True
            ).view(b, l, d_model)
        
        else: 
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5) 
            causal_mask = torch.tril(torch.ones(attn_logits.shape[-1], attn_logits.shape[-1]), diagonal=0).unsqueeze(0).unsqueeze(0).bool().to(self.device)
            attn_logits = attn_logits.masked_fill(causal_mask == 0, float("-inf"))
            attn_scores = F.softmax(attn_logits, dim=-1)
            attn_output = torch.matmul(attn_scores, v).view(b, l, d_model)  
        '''
        
        return attn_output

class MultiValueAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        context_len: int,
        ntk_rope_scaling: Union[dict, bool],
        dyn_scaling: Union[bool, float],
    ):
        """
        Initializes the MultiValueAttention module for attention with a single key head.

        Args:
            n_heads (int): Number of query and value attention heads.
            d_model (int): Model dimensionality (must be divisible by n_heads).
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool]): If dict, contains 'pretrained_context_window' and
                'new_context_window' for NTK scaling; if False, no scaling.
            dyn_scaling (Union[bool, float]): If float between 0 and 1, applies dynamic scaling; if None or False, no scaling.
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.causal_mask = torch.tril(torch.ones(self.context_len, self.context_len), diagonal=0).unsqueeze(0).unsqueeze(0).bool().to(self.device)

        self.rope = RotaryPositionalEmbedding(
            d_head=self.d_head,
            context_len=self.context_len,
            ntk_rope_scaling=self.ntk_rope_scaling,
            dyn_scaling=self.dyn_scaling
        )

    def forward(self, q, k, v, _inference=False):
        """
        Computes multi-value attention with a single key head and multiple query/value heads.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, d_head).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, d_model).
            _inference (bool, optional): If True, adjusts RoPE for inference mode. Defaults to False.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, sequence_length, d_model).
        """
        b, q_l, d_model = q.shape
        b, v_l, _ = v.shape 
        
        assert d_model == self.d_model, f"Expected d_model to be {self.d_model}, got {d_model}"

        if _inference:
            seq_len = q_l
        else:
            seq_len = self.context_len

        q = q.view(b, self.n_heads, q_l, self.d_head)
        k = k.unsqueeze(1)
        v = v.view(b, self.n_heads, v_l, self.d_head)

        if not _inference:
            assert q.shape[2:] == k.shape[2:], f"Expected q and k to match on seq_len and d_head dims, got {q.shape}, {k.shape}"
            assert q.shape == v.shape, f"Expected q and v to have the same shape, got {q.shape}, {v.shape}"

        q = self.rope(q, _inference=_inference, _q=True)
        k = self.rope(k, _inference=_inference)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :q_l, :v_l] == 0, float("-inf"))
        attn_scores = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_scores, v).view(b, seq_len, d_model)

        return attn_output
        
class GroupedQueryAttention(nn.Module):
    """
    Implements grouped query attention where heads are grouped, and each group shares keys and values.

    Attributes:
        n_heads (int): Number of attention heads.
        n_groups (int): Number of groups.
        d_model (int): Dimensionality of the input features.
        d_head (int): Dimensionality of each attention head (d_model // n_heads).
        context_len (int): Maximum sequence length.
        ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling.
        dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE.
    """
    
    def __init__(
        self,
        n_heads: int,
        n_groups: int,
        d_model: int, 
        context_len: int,
        ntk_rope_scaling: Union[dict, bool], 
        dyn_scaling: Union[bool, float],
        flash_attn:bool = False,
        flash_attn_dtype:torch.dtype = torch.float16
    ):
        """
        Initializes the GroupedQueryAttention module.

        Args:
            n_heads (int): Number of attention heads.
            n_groups (int): Number of groups.
            d_model (int): Dimensionality of the input features.
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool]): If dict, contains 'pretrained_context_window' and 'new_context_window'
                for NTK RoPE scaling; if False, no scaling.
            dyn_scaling (Union[bool, float]): If float between 0 and 1, applies dynamic scaling to RoPE; if False, no scaling.
            flash_attn (bool, optional): If True, uses FlashAttention for faster computation. Defaults to False.
            flash_attn_dtype (torch.dtype, optional): Data type for FlashAttention. Defaults to torch.float16.
        """
        super().__init__()
        self.n_heads = n_heads 
        self.n_groups = n_groups
        self.d_model = d_model
        self.d_head = d_model // n_heads 
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
        self.rope = RotaryPositionalEmbedding(
            d_head=self.d_head,
            context_len=self.context_len,
            ntk_rope_scaling=self.ntk_rope_scaling,
            dyn_scaling=self.dyn_scaling
        )
    
    def forward(self, q, k, v, _inference=False):
        """
        Computes grouped query attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, sequence_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, sequence_length, d_head * n_groups).
            v (torch.Tensor): Value tensor of shape (batch_size, sequence_length, d_head * n_groups).
            _inference (bool, optional): If True, adjusts RoPE for inference mode. Defaults to False.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, sequence_length, d_model).
        """
        b, l, d_model = q.shape
        _, l_k, _ = k.shape 
        _, l_v, _ = v.shape
       
        assert l_k == l_v, f"Expected k and v to have same sequence length, got {l_k}, {l_v}"  
        assert self.n_heads % self.n_groups == 0, f"Expected n_heads divisible by n_groups, got n_heads: {self.n_heads}, n_groups: {self.n_groups}"
        assert d_model == self.d_model, f"Expected d_model to be {self.d_model}, got {d_model}"  
       
        q = q.view(b, self.n_heads, l, self.d_head)
        k = k.view(b, self.n_groups, l_k, self.d_head)
        v = v.view(b, self.n_groups, l_v, self.d_head)
      
        if self.flash_attn:
            q = self.rope(q, _inference=_inference, _q=True).to(self.flash_attn_dtype)
            k = self.rope(k, _inference=_inference).to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype)

        attn_output = F.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            is_causal = True,
            enable_gqa = True
        ).contiguous().view(b, l, d_model).to(torch.float32)
 
        '''
        if self.flash_attn:
            q = self.rope(q, _inference=_inference, _q=True).to(self.flash_attn_dtype)
            k = self.rope(k, _inference=_inference).to(self.flash_attn_dtype)
            v = v.to(self.flash_attn_dtype)

            attn_output = F.scaled_dot_product_attention(
                query = q,
                key = k,
                value = v,
                is_causal = True,
                enable_gqa = True
            ).view(b, l, d_model)
        
        else:
            repeats = int(self.n_heads / self.n_groups)
            q = self.rope(q)
            k = self.rope(k.repeat_interleave(repeats=repeats, dim=1), _inference = _inference)
            v = v.repeat_interleave(repeats = repeats, dim = 1)

            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5) 
            causal_mask = torch.tril(torch.ones(attn_logits.shape[-1], attn_logits.shape[-1]), diagonal=0).unsqueeze(0).unsqueeze(0).bool().to(self.device)
            attn_logits = attn_logits.masked_fill(causal_mask == 0, float("-inf"))
            attn_scores = F.softmax(attn_logits, dim=-1)
            attn_output = torch.matmul(attn_scores, v).view(b, l, d_model)      
        '''
        return attn_output 

class TopKSparseVAttention(nn.Module):
    """
    An experimental implementation of sparse attention using the Top-K mechanism. KV Cache not Implemented.

    Args:
        n_heads (int): The number of attention heads.
        d_model (int): The model dimensionality (must be divisible by n_heads).
        top_k_sparsev (int): The number of top attention scores to retain per query.
        context_len (int): The maximum sequence length (context size).
        ntk_rope_scaling (Union[dict, bool]): If provided as a dictionary, should include
            'pretrained_context_window' and 'new_context_window' for NTK scaling; 
            if False, no scaling will be applied.
        dyn_scaling (Union[bool, float]): If set to a float between 0 and 1, applies
            dynamic scaling for RoPE. If False, no dynamic scaling is applied.
        flash_attn (bool, optional): If True, uses flash attention (not yet implemented).
        flash_attn_dtype (torch.dtype, optional): The data type for flash attention 
            (defaults to `torch.float16`). Flash attention requires this dtype to work optimally.

    Example:
        model = TopKSparseVAttention(n_heads=8, d_model=512, top_k_sparsev=8, context_len=1024, ntk_rope_scaling=True, dyn_scaling=0.1)
        q, k, v = torch.randn(2, 1024, 512), torch.randn(2, 1024, 512), torch.randn(2, 1024, 512)
        output = model(q, k, v)
    
    Note:
        This implementation is experimental and may undergo significant changes. It is also currently very slow, not recommend for use.
    """
    
    def __init__(
        self,
        n_heads: int, 
        d_model: int, 
        top_k_sparsev: int,
        context_len: int, 
        ntk_rope_scaling: Union[dict, bool], 
        dyn_scaling: Union[bool, float],
        flash_attn: bool = False,
        flash_attn_dtype: torch.dtype = torch.float16        
    ):
        super().__init__()

        self.n_heads = n_heads 
        self.d_model = d_model
        self.top_k_sparsev = top_k_sparsev
        self.d_head = d_model // n_heads 
        self.context_len = context_len
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
      
        assert self.top_k_sparsev is not None, 'top_k must not be None' 
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
        self.rope = RotaryPositionalEmbedding(
            d_head=self.d_head,
            context_len=self.context_len,
            ntk_rope_scaling=self.ntk_rope_scaling,
            dyn_scaling=self.dyn_scaling
        )

    def forward(self, q, k, v, _inference=False):
        """
        Performs the forward pass of the TopK sparse attention mechanism.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            _inference (bool, optional): If True, adjusts the RoPE for inference mode.
                Defaults to False.

        Returns:
            torch.Tensor: The attention output tensor of shape (batch_size, seq_length, d_model).
        
        """
        
        if _inference:
            raise NotImplementedError('_inference mode not implemented for TopKSparseVAttention')
        
        b, T, d_model = q.shape
       
        assert d_model == self.d_model
        assert q.shape == v.shape

        q = q.view(b, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)  
        v = v.view(b, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)  
        k = k.unsqueeze(1)  

        q = self.rope(q, _inference = _inference)
        k = self.rope(k, _inference = _inference)
        
        scale = 1.0 / math.sqrt(self.d_head)
        idx = torch.arange(T, device=q.device)
        causal = idx.unsqueeze(0) <= idx.unsqueeze(1) 
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale 
        attn_logits = attn_logits.masked_fill(~causal[None, None], float("-inf"))
        topk_vals, topk_idx = attn_logits.topk(self.top_k_sparsev, dim=-1) 
        attn_scores = F.softmax(topk_vals, dim=-1)  
        B, H, T, k = topk_idx.shape
        D = self.d_head
        batch_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1).expand(B, H, T, k)
        head_idx = torch.arange(H, device=q.device).view(1, H, 1, 1).expand(B, H, T, k)
        topk_v = v[batch_idx, head_idx, topk_idx] 
        out = (attn_scores.unsqueeze(-1) * topk_v).sum(dim=3) 

        return out.permute(0, 2, 1, 3).reshape(b, T, d_model)
 
class FeedForwardSwiGLU(nn.Module):
    """
    Implements a feed-forward network with SwiGLU activation.

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        swiglu_linear (nn.Linear): Linear layer for SwiGLU computation.
        swiglu_gate_linear (nn.Linear): Gate linear layer for SwiGLU.
        linear_out (nn.Linear): Output linear layer.
    """
    
    def __init__(self, d_model: int, h_dim:int):
        """
        Initializes the FeedForwardSwiGLU module.

        Args:
            d_model (int): Dimensionality of the input and output features.
        """
        super().__init__()
        self.d_model = d_model
        self.swiglu_linear = nn.Linear(d_model, h_dim)
        self.swiglu_gate_linear = nn.Linear(d_model, h_dim)
        self.linear_out = nn.Linear(h_dim, d_model)
        
    def forward(self, x):
        """
        Processes the input through the feed-forward network with SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        x = self.linear_out(x) 
        return x
   
class PositionalEmbedding(nn.Module):
    """
    Implements positional embeddings, either learned or fixed sinusoidal.

    Attributes:
        context_len (int): Maximum sequence length.
        learned (bool): Whether to use learned embeddings (True) or fixed sinusoidal (False).
        dropout (nn.Dropout): Dropout layer for regularization.
        positional_embedding (torch.Tensor or nn.Embedding): Positional embeddings buffer or layer.
    """
    
    def __init__(
        self, 
        context_len: int,
        d_model: int, 
        dropout_p: float = 0.1, 
        learned: bool = False
    ):
        """
        Initializes the PositionalEmbedding module.

        Args:
            context_len (int): Maximum sequence length.
            d_model (int): Dimensionality of the embeddings.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
            learned (bool, optional): If True, uses learned embeddings; if False, uses sinusoidal. Defaults to False.
        """
        super().__init__()
        self.context_len = context_len
        self.learned = learned
        self.dropout = nn.Dropout(p=dropout_p)  
        
        if not learned:
            pe = torch.zeros(size=(context_len, d_model), dtype=torch.float32)
            position = torch.arange(start=0, end=context_len, dtype=torch.float32)
            div_term = 10000 ** (torch.arange(start=0, end=d_model, step=2, dtype=torch.float32) / d_model)  
            pe[:, 0::2] = torch.sin(position.unsqueeze(1) / div_term)
            pe[:, 1::2] = torch.cos(position.unsqueeze(1) / div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("positional_embedding", pe)
        else:
            self.positional_embedding = nn.Embedding(context_len, d_model)
      
    def forward(self, x, _inference=False):
        """
        Adds positional embeddings to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            _inference (bool, optional): If True, handles inference mode with caching. Defaults to False.

        Returns:
            torch.Tensor: Input tensor with positional embeddings added.
        """
        
        if not _inference:
            x_pe = x + self.positional_embedding[:, :x.size(1), :] 
            x = self.dropout(x_pe) 
            return x  
        if _inference and (not hasattr(self, 't') or self.t is None):
            x_pe = x + self.positional_embedding[:, :x.size(1), :] 
            x = self.dropout(x_pe) 
            self.t = x.size(1)
            return x  
        elif _inference and (hasattr(self, 't') or self.t is not None):
            assert x.shape[1] == 1, f"Expected sequence length of 1 once cache exists, got {x.shape[1]}"
            x_pe = x + self.positional_embedding[:, self.t - 1, :]
            x = self.dropout(x_pe)
            if self.t >= self.context_len:
                self.t = self.context_len
            else:
                self.t += 1
            return x

    def _reset_cache(self, _t=True, _pe=False):
        """
        Resets the internal state for inference.

        Args:
            _t (bool, optional): If True, deletes the time step counter. Defaults to True.
            _pe (bool, optional): If True, deletes the positional embedding buffer (not typically used). Defaults to False.
        """
        if _t and hasattr(self, 't'):
            del self.t
        if _pe and hasattr(self, 'positional_embedding'):
            del self.positional_embedding
    
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements rotary positional embeddings with optional scaling for extended context windows.

    Attributes:
        context_len (int): Maximum sequence length.
        rope_cos (torch.Tensor): Cosine component of rotary embeddings.
        rope_sin (torch.Tensor): Sine component of rotary embeddings.
    """

    def __init__(
        self, 
        d_head: int, 
        context_len: int,
        ntk_rope_scaling: Union[dict, bool] = False, 
        dyn_scaling: Union[bool, float] = None
    ):
        """
        Initializes the RotaryPositionalEmbedding module.

        Args:
            d_head (int): Dimensionality of each attention head.
            context_len (int): Maximum sequence length.
            ntk_rope_scaling (Union[dict, bool], optional): If dict, contains 'pretrained_context_window' and
                'new_context_window' for NTK scaling; if False, no scaling. Defaults to False.
            dyn_scaling (Union[bool, float], optional): If float between 0 and 1, applies dynamic scaling; if None or False, no scaling. Defaults to None.
        """
        super().__init__()
        self.context_len = context_len
        position = torch.arange(start=0, end=context_len, dtype=torch.float16).unsqueeze(1)
        
        if ntk_rope_scaling:
            assert isinstance(ntk_rope_scaling, dict), "ntk_rope_scaling must be a dictionary"
            assert 'pretrained_context_window' in ntk_rope_scaling, "Missing 'pretrained_context_window' in ntk_rope_scaling"
            assert 'new_context_window' in ntk_rope_scaling, "Missing 'new_context_window' in ntk_rope_scaling"
            
            if dyn_scaling:
                assert isinstance(dyn_scaling, float), "dyn_scaling must be a float"
                assert 0 < dyn_scaling <= 1, "dyn_scaling must be between 0 and 1"
                scale = (dyn_scaling * (ntk_rope_scaling['new_context_window'] / ntk_rope_scaling['pretrained_context_window'])) + (1 - dyn_scaling)
            else:
                scale = ntk_rope_scaling['new_context_window'] / ntk_rope_scaling['pretrained_context_window']
            
            position /= scale

        div_term = 10000 ** (torch.arange(start=0, end=d_head, step=2, dtype=torch.float16) / d_head)
        div_term = torch.repeat_interleave(div_term, repeats=2, dim=-1)
        rope_cos = torch.cos(position / div_term)
        rope_sin = torch.sin(position / div_term)
        
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def forward(self, x, _inference=False, _q=False):
        """
        Applies rotary positional embeddings to the input.

        Args:
            x (torch.Tensor): Input tensor (queries or keys) of shape (batch_size, n_heads, sequence_length, d_head).
            _inference (bool, optional): If True, handles inference mode with position tracking. Defaults to False.
            _q (bool, optional): If True, input is queries; affects position tracking in inference. Defaults to False.

        Returns:
            torch.Tensor: Input with rotary positional embeddings applied.
        """
        if not _inference:
            cos = self.rope_cos[:x.shape[2]]  
            sin = self.rope_sin[:x.shape[2]]
        elif _inference and _q:
            if not hasattr(self, "t") or self.t is None:
                self.t = 1  
            else:
                self.t = min(self.t + 1, self.context_len)
            cos = self.rope_cos[self.t - 1:self.t]  
            sin = self.rope_sin[self.t - 1:self.t]
        elif _inference and not _q:
            cos = self.rope_cos[:x.shape[2]]
            sin = self.rope_sin[:x.shape[2]]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos_even = cos[..., 0::2]
        sin_even = sin[..., 0::2]

        rotated_x_even = x_even * cos_even - x_odd * sin_even
        rotated_x_odd = x_even * sin_even + x_odd * cos_even

        rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        rotated = rotated.flatten(-2)
        return rotated