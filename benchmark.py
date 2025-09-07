import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import argparse
import math
import os
from nmdl import TransformerLM


def get_model_and_tokenizer(model_path, device, context_len=512):
    # Load tokenizer (match training config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.add_special_tokens({'pad_token': '<p>'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<think>', '</think>', '<doc>', '</doc>']})
    vocab_size = len(tokenizer)
    
    def collect_global_token_ids(tokenizer):
        """
        Pick a conservative set of global tokens: BOS and domain markers if present.
        """
        global_ids = set()
        if tokenizer.bos_token_id is not None:
            global_ids.add(int(tokenizer.bos_token_id))
        # Optional: treat '<doc>' and '<think>' as global anchors if added
        for tok in ['<doc>', '<think>']:
            tid = tokenizer.convert_tokens_to_ids(tok) if hasattr(tokenizer, "convert_tokens_to_ids") else None
            if tid is not None and tid != tokenizer.unk_token_id:
                global_ids.add(int(tid))
        return sorted(list(global_ids))



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

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
        print("Model weights loaded successfully!")
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Model will use random weights.")
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark TheTransformer on wikitext-002")
    parser.add_argument("--model", type=str, default="weights/large4.pth", help="Path to model weights file")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for evaluation")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model, tokenizer = get_model_and_tokenizer(args.model, device, context_len=args.context_len)

    print("Loading wikitext-002 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"Loaded {len(dataset)} samples.")

    def preprocess(example):
        text = example["text"].strip()
        if not text:
            return {"input_ids": []}
        return {"input_ids": tokenizer.encode(text, add_special_tokens=False,padding = False, truncation= False)}

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 1)

    # Group into blocks of context_len
    def group_texts(examples):
        block_size = args.context_len
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        result = {
            "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
        }
        return result

    dataset = dataset.map(group_texts, batched=True, batch_size=1000)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) == args.context_len)

    # Prepare DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Evaluating on {len(dataset)} blocks of {args.context_len} tokens...")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, targets=labels)


            total_loss += loss.item()
            if i % 20 == 0:
                print(f"Step {i}: Loss={loss.item():.4f} - Perplexity={math.exp(loss.item())}")

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
    print(f"\nFinal Results on wikitext-002:")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
