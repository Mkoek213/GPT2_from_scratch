import math
import numpy as np
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

