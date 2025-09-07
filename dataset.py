from datasets import load_dataset, interleave_datasets

class StreamedDataset():
    def __init__(self, tokenizer, split, context_len):
        assert split in ["train", "validation", "test"], "Invalid split provided. Choose from 'train', 'validation', 'test'."
        self.tokenizer = tokenizer
        self.context_len = context_len

        mixture_proportions = {
            "fineweb": 0.60,
            "the_stack": 0.20,
            "books3": 0.10,
            "arxiv": 0.05,
            "slim_orca": 0.05,
        }

        fineweb_dataset = self.keepContent(load_dataset(
            "HuggingFaceFW/fineweb",
            split=split,
            streaming=True))

            
        stack_dataset = self.keepContent(load_dataset(
            'bigcode/the-stack-dedup',
            data_dir='data',
            split=split,
            streaming=True
        ))

        books3_dataset = self.keepContent(load_dataset(
            'ArmelR/the-pile-splitted',
            data_dir="data/Books3",
            split=split,
            streaming=True,
        ).rename_column("text", "content"))


        arxiv_dataset = self.keepContent(load_dataset(
            'ArmelR/the-pile-splitted',
            data_dir="data/ArXiv",
            split=split,
            streaming=True,
        ).rename_column("text", "content"))


        slim_orca_dataset = self.keepContent(load_dataset(
            'Open-Orca/SlimOrca-Dedup',
            split=split,
            streaming=True,
        ).rename_column("conversations", "content")).map(self.parse_conversation, batched=True)


        self.data = interleave_datasets(
            datasets=[
                fineweb_dataset,
                stack_dataset,
                books3_dataset,
                arxiv_dataset,
                slim_orca_dataset,
            ],
            probabilities=[
                mixture_proportions["fineweb"],
                mixture_proportions["the_stack"],
                mixture_proportions["books3"],
                mixture_proportions["arxiv"],
                mixture_proportions["slim_orca"],
            ],
            seed=42,
            stopping_strategy="all_exhausted"
        )

        self.data = self.data.map(self.tokenize_fn, batched=True, remove_columns=["content"]).map(self.group_texts, batched=True, fn_kwargs={'block_size': self.context_len})

    def parse_conversation(self, example):
        new_batch = []
        for batch in example['content']:
            complete_message = ""
            for message in batch:
                if isinstance(message, dict) and 'from' in message and 'value' in message:
                    complete_message += f"\n{message['from'].upper()}: {message['value']}"
            if complete_message.strip():
                new_batch.append(complete_message.strip())
            else:
                new_batch.append("")
        
        return {"content": new_batch}



    def tokenize_fn(self, entry):
        examples = []
        for item in entry["content"]:
            if isinstance(item, str) and item is not None and item.strip():
                examples.append(item.strip())
            else:
                examples.append("")

        tokenized = self.tokenizer(
            examples,
            truncation=False,
            padding=False, 
            return_tensors=None, 
            add_special_tokens=False
        )
        
        return tokenized

    
    def group_texts(self, tokenized_dataset, block_size=None):
        if block_size is None:
            block_size = self.context_len
        
        effective_block_size = block_size - 2
        
        all_input_ids = []
        all_attention_masks = []

        for doc_ids, attention_mask in zip(tokenized_dataset['input_ids'], tokenized_dataset['attention_mask']):
            if doc_ids and len(doc_ids) > 0:
                doc_with_special_tokens = [self.tokenizer.bos_token_id] + doc_ids + [self.tokenizer.eos_token_id]
                attention_with_special = [1] + attention_mask + [1]
                
                all_input_ids.append(doc_with_special_tokens)
                all_attention_masks.append(attention_with_special)

        concatenated_ids = []
        concatenated_masks = []
        
        for doc_ids, doc_masks in zip(all_input_ids, all_attention_masks):
            concatenated_ids.extend(doc_ids)
            concatenated_masks.extend(doc_masks)

        result = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for i in range(0, len(concatenated_ids) - block_size + 1, effective_block_size):
            block_ids = concatenated_ids[i:i + block_size]
            block_masks = concatenated_masks[i:i + block_size]
            
            if len(block_ids) == block_size:
                result['input_ids'].append(block_ids)
                result['attention_mask'].append(block_masks)
                
        return result



    
    def keepContent(self, dataset):
        """
        Keep only the 'content' column from the dataset.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Dataset with only 'content' column
        """
        columns_to_remove = [col for col in dataset.column_names if str(col) != "content"]
        if columns_to_remove:
            return dataset.remove_columns(columns_to_remove)
        return dataset


    def getData(self):
        return self.data

