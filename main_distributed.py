import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.optim.lr_scheduler import _LRScheduler

import math
import time

from nmdl import TransformerLM
from dataset import StreamedDataset

from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from datasets.distributed import split_dataset_by_node

import torch.distributed as dist
import os

#--------------------------------------------------
# Data Preparation
#--------------------------------------------------




def setup_ddp():
    dist.init_process_group(backend="nccl" )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

#--------------------------------------------------
# Main Training Function
#--------------------------------------------------





from torch.utils.data import get_worker_info

class HFIterableDataset(IterableDataset):
    def __init__(self, hf_streamed_dataset):
        self.hf_streamed_dataset = hf_streamed_dataset

    def __iter__(self):
        worker = get_worker_info()
        it = iter(self.hf_streamed_dataset)
        if worker is None:
            return it  # single-process
        # Simple strided sharding: worker.id, num_workers
        # Works for infinite/long streams, but note: with HF shuffle buffers,
        # perfect randomness is approximate.
        import itertools
        return itertools.islice(it, worker.id, None, worker.num_workers)





def main():
    rank, local_rank = setup_ddp()
    dist.barrier()

    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.add_special_tokens({'pad_token': '<p>'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<think>', '</think>', '<doc>', '</doc>']})
    vocab_size = len(tokenizer)
    context_len = 2048
    
    dataset = StreamedDataset(tokenizer=tokenizer, split='train', context_len=context_len)
    dataset = dataset.getData().shuffle(seed=42, buffer_size=10_000)

    
    
    
    dataset = split_dataset_by_node(dataset, rank=int(rank), world_size=int(dist.get_world_size()))




    torch_dataset = HFIterableDataset(dataset)
    
    collate = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    train_dataloader = DataLoader(
        torch_dataset,
        batch_size=8,
        collate_fn=collate,
        num_workers=2,
        multiprocessing_context=ctx,   # <— important
        persistent_workers=True,       # fewer forks during training
        prefetch_factor=2,             # start conservative
        pin_memory=True
    )

    if rank == 0:
        example = next(iter(train_dataloader))

    

    model = TransformerLM(
        vocab_size=len(tokenizer),  
        dim=3072,   
        n_heads=32, 
        n_layers=32,         
        mlp_expansion=3.5,   
        dropout=0.1,         
        max_seq_len=4096,    
        use_rotary=True,     
        use_alibi=True,      
        ignore_index=-100    
    ).to(device)
    
    load_cp = False
    if load_cp:
        checkpoint = torch.load(r"weights/12ppl.pth", weights_only=True, map_location=torch.device(device))
        
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)

    model = torch.compile(model)

    scaler = torch.amp.GradScaler()
    dist.barrier()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    

    for param in model.parameters():
        param.requires_grad = True

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    base_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95))

    class CosineAnnealingWithWarmup(_LRScheduler):
        def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.min_lr = min_lr
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            step = self.last_epoch + 1 

            if step <= self.warmup_steps:
                # Linear warmup
                return [
                    base_lr * step / max(1, self.warmup_steps)
                    for base_lr in self.base_lrs
                ]
            else:
                # Cosine annealing
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                return [
                    self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs
                ]

    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_steps=2000,
        total_steps=610_352,
        min_lr=1e-6
    )
    n_epochs = 2
    

    t1=time.time()
    print('Training loop started.')
    for epoch in range(n_epochs):
        model.train()
        
        running_loss = 0
        total_tokens = 0


        print(f"Beginning epoch {epoch}.")
        for index, element in enumerate(train_dataloader):
            try:
                optimizer.zero_grad()
                
                text = element["input_ids"].to(device)
                label = element["labels"].to(device)
                attention_mask = element["attention_mask"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(idx=text, targets=label)

                is_finite_local = torch.tensor(float(torch.isfinite(loss)), device=device)
                dist.all_reduce(is_finite_local, op=dist.ReduceOp.MIN)

                if is_finite_local.item() == 0.0:
                    if rank == 0:
                        print("Non-finite loss detected on at least one rank; skipping step on ALL ranks")
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue


                non_padded_tokens = (label != tokenizer.pad_token_id).sum().item()
                running_loss += loss.item() * non_padded_tokens
                total_tokens += non_padded_tokens

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if rank == 0 and index % 100 == 0:
                    avg_loss = running_loss / total_tokens
                    perplexity = math.exp(avg_loss) if avg_loss < 50 else None
                    t2 = time.time()
                    
                    clr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}, Step {index}, LR: {clr}, Loss: {avg_loss}, Perplexity: {perplexity} - time: {(t2-t1)/60}")


                if rank == 0 and index % 1000 == 0:
                    saved_dict = model.module.state_dict()
                    torch.save(saved_dict, "weights/large4.pth")
                    train_dict = {"optim": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(),
                                  "steps": index}
                    torch.save(train_dict, "weights/cp.pth")
                    del saved_dict
                    del train_dict
                    print("Model saved")
            
            except:
                saved_dict = model.module.state_dict()
                torch.save(saved_dict, "weights/backup.pth")
                print("Model emergency save")
                continue

    cleanup_ddp()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()