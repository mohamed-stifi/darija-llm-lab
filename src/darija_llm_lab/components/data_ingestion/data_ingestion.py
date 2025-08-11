from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from darija_llm_lab.entity.config_entity import DAPTDataConfig, SFTDataConfig

class DataIngestion:
    def load_dapt_dataset(self, config: DAPTDataConfig, tokenizer):
        """Loads and prepares the raw text dataset for DAPT."""
        print("--- Loading and preparing DAPT dataset ---")
        split = "train"
        if config.debug_mode:
            split += f"[:{config.debug_split_size}]"
            
        dataset = load_dataset(config.dataset_id, split=split)
        
        # If a Processor (e.g., Gemma3nProcessor) was passed, use its .tokenizer
        tok = getattr(tokenizer, "tokenizer", tokenizer)

        # Safely get model_max_length; let HF handle default if missing/None
        max_len = getattr(tok, "model_max_length", None)

        def tokenize_function(examples):
            return tok(
                examples[config.text_column],
                truncation=True,
                max_length=max_len,   # ok if None; HF will use default
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        print(f"DAPT dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def load_sft_dataset(self,config: SFTDataConfig, tokenizer):
        """Loads, splits, filters, and formats the dataset."""
        print("\n--- Step 4: Loading and Preparing Data ---")
        split = "train"
        if config.debug_mode:
            split += f"[:{config.debug_split_size}]"
        
        dataset = load_dataset(config.dataset_id, split=split)
        
        split_dataset = dataset.train_test_split(
            test_size=config.test_size, seed=config.seed
        )
        train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
        # Filter dataset before applying template
        eval_dataset = eval_dataset.filter(lambda x: self._is_valid_conversation(x["messages"]))
        train_dataset = train_dataset.filter(lambda x: self._is_valid_conversation(x["messages"]))

        # Filter and transform data
        train_dataset = train_dataset.map(self._transform_message_format)
        eval_dataset = eval_dataset.map(self._transform_message_format)

        # Standardize format for Unsloth
        train_dataset = standardize_data_formats(train_dataset)
        eval_dataset = standardize_data_formats(eval_dataset)
        
        # Apply chat template
        train_dataset = train_dataset.map(lambda batch: self._format_with_chat_template(batch, tokenizer), batched=True)
        eval_dataset = eval_dataset.map(lambda batch: self._format_with_chat_template(batch, tokenizer), batched=True)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(eval_dataset)}")
        return train_dataset, eval_dataset

    def _transform_message_format(self, example):
        """Converts messages to the Gemma-3N required format."""
        return {
            "messages": [
                {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                for msg in example["messages"]
            ]
        }
    
    def _is_valid_conversation(self, conversation):
        # Must start with "user" and alternate with "assistant"
        if not conversation or conversation[0]["role"] != "user":
            return False
        expected = "user"
        for msg in conversation:
            if msg["role"] != expected:
                return False
            expected = "assistant" if expected == "user" else "user"
        return True

    def _format_with_chat_template(self, batch, tokenizer):
        """Applies chat template and removes the initial BOS token."""
        formatted_texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            ).removeprefix('<bos>')
            for conv in batch["messages"]
        ]
        return {'text': formatted_texts}

    