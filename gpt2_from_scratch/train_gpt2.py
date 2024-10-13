import os
from doctest import master
from torch.nn import functional as F
import tiktoken
import torch.distributed as dist
import torch
from torch.distributed._composable.replicate import DDP
import time
from torch.distributed import init_process_group, destroy_process_group
from gpt2_from_scratch.DataLoader import DataLoaderLite
from gpt2_from_scratch.GPT import GPT
from gpt2_from_scratch.GPTConfig import GPTConfig
from gpt2_from_scratch.hellaswag import render_example, iterate_examples
from gpt2_from_scratch.helper import get_lr, get_most_likely_row

# how to launch using ddp:
# torchrun --standalone --nproc_per_node=1 train_gpt2.py // --nproc_per_node=1, because in my case i have only 1 GPU

# set up ddp (distributed data parallel)
ddp = int(os.environ.get('RANK', -1)) not in {-1, 0} # not using ddp if you have none or one GPU
if ddp:
    print("DDP działa")
    assert torch.cuda.is_available(), init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # single gpu
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 1 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size ) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) # I can not do it on my GPU, it's too old
if ddp and ddp_local_rank >= 1: # no case of using ddp if you have only one GPU
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp and ddp_local_rank >= 1 else model

max_lr = 3 * 6e-4
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 1500
time_limit = 11 * 60 * 60 # in seconds, 11hours

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open so file will start empty
    pass

enc = tiktoken.get_encoding("gpt2")

start_time = time.time()



for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evaluate validation loss once per 20 steps
    if step % 20 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 4
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step} | validation loss: {val_loss_accum.item():.4f}\n")
                # write checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model_state_dict': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'state_dict': optimizer.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)


    # evaluate hellaswag once per 20 steps
    if step % 20 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step} | HellaSwag accuracy: {acc_norm:.4f}\n")



    # generate from the model once per 20 steps
    if (step > 0 and step % 20 == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print generated text:
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # turn on only when it is the last step
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set learning rate for this iteration
    lr = get_lr(iteration=step, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # in second
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    if master_process:
        print(f"step {step}| loss: {loss_accum.item():.6f}| lr: {lr:.4e}| norm: {norm:.4f} | dt: {dt:.2f}ms| tok/sec: {tokens_per_sec}")
        with open(log_file, "a") as f:
            f.write(f"step {step} | loss: {loss_accum.item():.6f}")
    elapsed_time = time.time() - start_time
    if elapsed_time >= time_limit:
        print(f"Training stopped after {elapsed_time} seconds.")
        break
if ddp:
    destroy_process_group()
