import torch.nn as nn
from .CausalSelfAttention import CausalSelfAttention
from .MLP import MLP


class Block(nn.Module):
    """
    A single transformer block that contains a layer normalization, a multi-head causal self-attention mechanism,
    and a feed-forward multilayer perceptron (MLP) layer. This block is the core building block of the transformer
    architecture, often stacked multiple times to form the model.

    The Block uses residual connections around the attention and MLP sublayers, allowing the model to learn better
    long-term dependencies in sequential data.

    Attributes:
    -----------
    ln_1 : nn.LayerNorm
        The first layer normalization applied before the attention mechanism.
    attn : CausalSelfAttention
        The multi-head causal self-attention mechanism.
    ln_2 : nn.LayerNorm
        The second layer normalization applied before the MLP (feed-forward) network.
    mlp : MLP
        A feed-forward MLP network applied after the attention mechanism.
    """

    def __init__(self, config):
        """
        Initializes the Block module.

        Parameters:
        -----------
        config : object
            Configuration object containing the following attributes:
            - n_embd: int, the size of the embedding vector (the dimensionality of the input and output).
        """
        super().__init__()

        # First layer normalization before applying attention
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Multi-head causal self-attention mechanism
        self.attn = CausalSelfAttention(config)

        # Second layer normalization before applying the feed-forward network
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed-forward MLP network (typically consists of two linear layers with activation in between)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass of the transformer block.

        The input first goes through layer normalization, then through the self-attention mechanism, with a residual
        connection that adds the input back to the output. The result is passed through another layer normalization
        and then through the MLP network, again with a residual connection.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and
            C is the embedding dimension.

        Returns:
        --------
        torch.Tensor
            The output tensor after passing through the attention mechanism, normalization, and MLP layers,
            with the same shape as the input (B, T, C).
        """
        # First, apply layer normalization and self-attention, then add the residual connection (input + output)
        x = x + self.attn(self.ln_1(x))  # Residual connection after self-attention

        # Apply layer normalization and MLP, then add the residual connection (input + output)
        x = x + self.mlp(self.ln_2(x))  # Residual connection after MLP

        return x
