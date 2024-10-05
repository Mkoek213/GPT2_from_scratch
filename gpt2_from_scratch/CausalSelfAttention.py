import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Implements a multi-head causal self-attention mechanism, as used in transformer models like GPT.

    This attention mechanism ensures that the model can only attend to past tokens in the sequence, not future ones,
    hence the term 'causal'. It includes key, query, and value projections for each attention head, as well as
    regularization and masking to preserve causality.

    Attributes:
    -----------
    c_attn : nn.Linear
        Linear layer for projecting the input into query, key, and value vectors.
    c_proj : nn.Linear
        Linear layer for projecting the output of the attention mechanism back to the embedding space.
    n_head : int
        The number of attention heads.
    n_embd : int
        The size of the embedding vector (also the input/output size of the projection layers).
    bias : torch.Tensor
        A causal mask that prevents attention from accessing future tokens. It is an upper-triangular matrix.
    """

    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.

        Parameters:
        -----------
        config : object
            Configuration object containing the following attributes:
            - n_embd: int, the size of the embedding vector.
            - n_head: int, the number of attention heads.
            - block_size: int, the maximum sequence length (used for masking).
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by the number of heads."

        # Linear layer to project input embeddings into queries, keys, and values for all heads in a single step
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Linear layer for output projection after attention is applied
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Custom initialization value for scaling (specific to implementation)

        # Number of attention heads and the embedding size
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask to prevent attention to future tokens. It is an upper triangular matrix with ones in the lower part.
        # This mask ensures that attention respects the causal order.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Forward pass of the CausalSelfAttention mechanism.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and
            C is the embedding dimension (n_embd).

        Returns:
        --------
        torch.Tensor
            The output tensor after applying self-attention and projecting back to the embedding space, with the
            same shape as input (B, T, C).
        """
        B, T, C = x.size()  # B: Batch size, T: Sequence length, C: Embedding dimensionality (n_embd)

        # Compute query (q), key (k), and value (v) vectors for all heads in one step using a linear projection.
        # The projection has dimensions (B, T, 3*C) where C is the embedding size. We split this into 3 parts:
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Split into query (q), key (k), and value (v), each of size (B, T, C)

        # Reshape and transpose to handle multiple attention heads.
        # Reshaped to (B, T, n_head, head_size), then transposed to (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Key: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Query: (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Value: (B, n_head, T, head_size)

        # Apply scaled dot-product attention using the built-in PyTorch function.
        # `is_causal=True` ensures that the attention is causal, meaning no attention to future tokens.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # After applying attention, reshape the output back to (B, T, C) by transposing and then merging the heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply the final output projection layer
        y = self.c_proj(y)

        return y
