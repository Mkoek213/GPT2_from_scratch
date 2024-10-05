import torch
import tiktoken


class DataLoaderLite:
    """
    A lightweight data loader that processes tokenized text data and generates mini-batches for training.

    This class is designed for distributed processing with multiple processes, each accessing a specific part
    of the tokenized text data. It reads tokenized data from disk, divides the data into batches of size B (batch size)
    and sequence length T (tokens per batch), and provides an efficient method for sequential data access across processes.

    Attributes:
    -----------
    B : int
        Batch size, the number of sequences (rows) in each batch.
    T : int
        Sequence length, the number of tokens in each sequence.
    process_rank : int
        The rank (ID) of the current process in a multi-process setup, used to determine data access for this process.
    num_processes : int
        The total number of processes that are dividing the data among themselves.

    Methods:
    --------
    next_batch():
        Returns the next batch of tokenized data for this process. Resets to the beginning if the end of data is reached.
    """

    def __init__(self, B, T, process_rank, num_processes):
        """
        Initializes the DataLoaderLite object.

        Parameters:
        -----------
        B : int
            Batch size, the number of sequences (rows) in each batch.
        T : int
            Sequence length, the number of tokens in each sequence.
        process_rank : int
            The rank (ID) of the current process in a multi-process setup.
        num_processes : int
            The total number of processes that are dividing the data.
        """
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.process_rank = process_rank  # Current process rank
        self.num_processes = num_processes  # Total number of processes

        # Load tokens from disk and store in memory
        with open('../input.txt', 'r') as f:
            text = f.read()  # Read the entire text file into memory
        enc = tiktoken.get_encoding('gpt2')  # Use GPT-2 encoding to tokenize the text
        tokens = enc.encode(text)  # Convert text to tokens
        self.tokens = torch.tensor(tokens)  # Convert tokens to a PyTorch tensor for easy manipulation
        print(f"loaded {len(self.tokens)} tokens")  # Output the total number of tokens loaded

        # Set the initial position for the current process, based on the batch size, sequence length, and process rank
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Generate and return the next batch of tokenized data.

        The data is returned as two tensors: `x` (input) and `y` (target), both shaped as (B, T). The target `y` is
        shifted by one token compared to the input `x`, which is a common setup for sequence prediction tasks.

        Returns:
        --------
        tuple:
            x : torch.Tensor
                Input tokens of shape (B, T), where B is the batch size and T is the sequence length.
            y : torch.Tensor
                Target tokens (next token in sequence), shifted by one token, of shape (B, T).
        """
        B, T = self.B, self.T  # Get batch size and sequence length
        # Extract a buffer of tokens starting from the current position
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # Inputs (B, T) - all except the last token in the buffer
        y = buf[1:].view(B, T)  # Targets (B, T) - all except the first token, shifted by one

        # Advance the current position by B * T tokens
        self.current_position += B * T

        # If the current position exceeds the available tokens for the current process, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # Reset the position to the start for the next epoch or when all data has been processed
            self.current_position = self.B * self.T * self.num_processes

        # Return the input and target tensors
        return x, y
