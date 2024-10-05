import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt2_from_scratch.Block import Block
from gpt2_from_scratch.GPTConfig import GPTConfig


class GPT(nn.Module):
    """
    GPT (Generative Pretrained Transformer) model class.
    This class implements a GPT model with transformer blocks, token and positional embeddings,
    and a final linear layer for language modeling. The model can be trained from scratch or
    initialized with pretrained weights.

    Args:
    -----
    config (GPTConfig): The configuration object that contains the model hyperparameters.
    """

    def __init__(self, config):
        """
        Initialize the GPT model with token and positional embeddings, transformer blocks,
        layer normalization, and a language modeling head. The token and positional embeddings
        are shared with the language model head for efficiency.

        Args:
        -----
        config (GPTConfig): Configuration object that provides hyperparameters such as
                            number of layers, embedding size, and vocabulary size.
        """
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: Maps input token IDs to dense vectors of size (vocab_size, n_embd)
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings: Maps positions in the sequence to dense vectors of size (block_size, n_embd)
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # Transformer blocks: List of Block objects (each containing attention and MLP layers)
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization
            ln_f=nn.LayerNorm(config.n_embd)
        ))

        # Final linear layer for language modeling head (maps from n_embd to vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Share weights between token embeddings and language model head (efficiency trick)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model using normal distribution for linear and embedding layers.
        Applies custom scaling for some linear layers depending on a flag.

        Args:
        -----
        module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Apply custom scaling if the module has the 'NANOGPT_SCALE_INIT' attribute
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= 2 * self.config.n_layer ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        """
        Perform a forward pass through the model.

        Args:
        -----
        idx (torch.Tensor): Input tensor of shape (B, T), where B is the batch size, and T is the sequence length.
        targets (torch.Tensor, optional): Target tensor for language modeling task. Default is None.

        Returns:
        --------
        logits (torch.Tensor): Output logits of shape (B, T, vocab_size).
        loss (torch.Tensor, optional): Cross-entropy loss if targets are provided, else None.
        """
        B, T = idx.size()  # B: batch size, T: sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Generate position indices for positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Get position and token embeddings
        pos_emb = self.transformer.wpe(pos)  # Shape: (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # Shape: (B, T, n_embd)
        x = tok_emb + pos_emb  # Add token and position embeddings

        # Pass through each transformer block
        for block in self.transformer.h:
            x = block(x)

        # Apply final layer normalization
        x = self.transformer.ln_f(x)
        # Output logits for each token position (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Compute cross-entropy loss between logits and targets (flattened)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load a pretrained GPT-2 model from HuggingFace and align its weights with the current GPT model.

        Args:
        -----
        model_type (str): The type of pretrained GPT-2 model to load.
                          Must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.

        Returns:
        --------
        model (GPT): GPT model with pretrained weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT-2 model: {model_type}")

        # Set configuration based on model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # Fixed vocab size for GPT-2 models
        config_args['block_size'] = 1024  # Fixed sequence length for GPT-2 models

        # Create a GPT model from scratch with the corresponding configuration
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Load HuggingFace model state dict and match it with the custom GPT model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Match and transpose weights as needed, especially for linear layers
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Ensure all parameters match in shape and copy from HuggingFace model
        sd = model.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape  # Transpose linear layers
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape  # Direct copy
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        Configure the AdamW optimizer for training, with separate weight decay for different parameter groups.

        Args:
        -----
        weight_decay (float): The weight decay value for regularization.
        learning_rate (float): The learning rate for optimization.
        device (str): The device type ('cuda' or 'cpu').

        Returns:
        --------
        optimizer (torch.optim.Optimizer): The AdamW optimizer.
        """
        # Filter out non-trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Separate parameters into groups with or without weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # Decayed (weights of layers and embeddings)
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # Non-decayed (biases, LayerNorms)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Print parameter count info
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")

        # Use fused AdamW optimizer if available and on CUDA device
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")

        # Return the configured optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
