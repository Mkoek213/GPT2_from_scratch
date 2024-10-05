import torch.nn as nn


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) block used within transformer architectures. This block
    consists of a feed-forward network with two fully connected layers, a GELU activation
    function, and projection layers for dimensionality transformation.

    Parameters:
    -----------
    config : object
        Configuration object containing model hyperparameters, specifically `n_embd`,
        which represents the embedding size.
    """

    def __init__(self, config):
        """
        Initializes the MLP block with two linear layers and a GELU activation.

        Parameters:
        -----------
        config : object
            A configuration object containing the model's hyperparameters. The `n_embd`
            attribute is used to set the size of the input embedding and the internal
            dimensions of the linear layers.
        """
        super().__init__()
        # First fully connected layer: projects from the embedding size to 4x the embedding size
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU activation function with an approximation method
        self.gelu = nn.GELU(approximate='tanh')

        # Second fully connected layer: projects back to the original embedding size
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # Custom attribute used for specific initialization in NanoGPT models
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Custom scaling initialization for specific use

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
        --------
        torch.Tensor
            The output tensor after passing through the MLP block.
        """
        # Pass input through the first fully connected layer
        x = self.c_fc(x)

        # Apply GELU activation
        x = self.gelu(x)

        # Pass through the second fully connected layer to project back to the original embedding size
        x = self.c_proj(x)

        return x
