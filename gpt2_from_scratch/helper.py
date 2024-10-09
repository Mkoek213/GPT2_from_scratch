import math
import numpy as np
from torch.nn import functional as F
import torch


def get_lr(iteration, warmup_steps, max_steps, max_lr, min_lr):
    """
    Calculates the learning rate for a given iteration based on a warmup and cosine decay schedule.

    Parameters:
    -----------
    iteration : int
        The current iteration or step of the training process.
    warmup_steps : int
        Number of steps for the linear warmup phase, where the learning rate increases from 0 to max_lr.
    max_steps : int
        The total number of steps after which the learning rate will remain at the minimum value (min_lr).
    max_lr : float
        The maximum learning rate, achieved at the end of the warmup phase.
    min_lr : float
        The minimum learning rate, which the learning rate decays to after max_steps.

    Returns:
    --------
    float
        The calculated learning rate for the current iteration.
    """

    # 1) Linear warmup phase: for iterations less than the warmup_steps,
    #    the learning rate increases linearly from 0 to max_lr.
    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps

    # 2) After the maximum steps, the learning rate is clamped to the minimum value (min_lr).
    if iteration > max_steps:
        return min_lr

    # 3) Cosine decay phase: for iterations between warmup_steps and max_steps,
    #    the learning rate decreases following a cosine decay curve.
    decay_ratio = (iteration - warmup_steps) / (
                max_steps - warmup_steps)  # Calculate the decay ratio based on the current iteration.
    assert 0 <= decay_ratio <= 1  # Ensure the decay ratio is within valid bounds (0 to 1).

    # Cosine decay: the coefficient starts at 1 and decreases to 0 as the iteration progresses.
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay function.

    # Final learning rate is interpolated between max_lr and min_lr based on the cosine decay coefficient.
    return min_lr + coeff * (max_lr - min_lr)


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

