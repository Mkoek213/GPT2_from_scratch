from attr import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT (Generative Pretrained Transformer) models.
    This class holds all the necessary hyperparameters for defining the architecture
    of a GPT model. These parameters are used to configure the model's layers, attention
    heads, and embedding size, as well as the tokenization settings like vocabulary size
    and sequence length.

    Attributes:
    -----------
    block_size : int
        The maximum sequence length (number of tokens) that the model can process in a single input.
        Default is 1024.

    vocab_size : int
        The number of tokens in the vocabulary. This is a combination of:
        - 50,000 Byte Pair Encoding (BPE) tokens
        - 256 byte tokens (to cover raw bytes)
        - 1 special token (End-of-File or EOF token)
        Default is 50257.

    n_layer : int
        The number of transformer layers (blocks) in the model. Each layer consists of a multi-head
        self-attention mechanism and a feed-forward network. Default is 12 layers.

    n_head : int
        The number of attention heads in each multi-head self-attention mechanism. Each head learns
        different parts of the input to improve the model’s representation capability. Default is 12 heads.

    n_embd : int
        The dimensionality of the embeddings used in the model. This defines the size of the input and
        output vectors for each layer and determines how rich or detailed the token representations are.
        Default is 768.
    """

    block_size: int = 1024  # Maximum sequence length (number of tokens) that the model can handle
    vocab_size: int = 50257  # Size of the vocabulary: 50,000 BPE tokens + 256 byte tokens + 1 EOF token
    n_layer: int = 12  # Number of transformer layers in the model
    n_head: int = 12  # Number of attention heads in each self-attention block
    n_embd: int = 768  # Dimensionality of token embeddings (and the hidden size of the model)
