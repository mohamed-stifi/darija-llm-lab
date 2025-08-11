import torch
import json
import os
from unsloth import FastModel
from transformers import GemmaTokenizerFast
from tokenizers import Tokenizer
from darija_llm_lab.entity.config_entity import ModelConfig, SFTPEFTConfig
from darija_llm_lab.utils.common import print_trainable_parameters

class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, model_path: str = None):
        """Loads model and tokenizer from HF hub or a local path."""
        load_path = model_path if model_path else self.config.base_model_id
        print(f"--- Loading model from: {load_path} ---")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=load_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            full_finetuning = False, 
        )
        return self.model, self.tokenizer

    def perform_embedding_surgery(self):
        """
        Adds new tokens to the model's vocabulary by resizing and shifting
        the embedding and lm_head layers, ensuring multimodal tokens are preserved.
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        print("\n--- Step 2: Performing Embedding Surgery ---")
        
        # --- Stage 1: Model Embedding Resizing ---
        original_vocab_size = self.model.config.text_config.vocab_size
        input_embeddings = self.model.get_input_embeddings()
        lm_head = self.model.get_output_embeddings()
        
        text_boundary_idx = self.model.config.boi_token_id
        num_new_tokens = len(self.surgery_config.new_special_tokens)
        new_vocab_size = original_vocab_size + num_new_tokens
        
        print(f"Adding {num_new_tokens} new tokens. New vocabulary size will be: {new_vocab_size}")

        with torch.no_grad():
            new_input_embeddings = self._create_new_embedding_matrix(input_embeddings, text_boundary_idx, num_new_tokens, new_vocab_size)
            new_lm_head = self._create_new_embedding_matrix(lm_head, text_boundary_idx, num_new_tokens, new_vocab_size)

        self.model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8, mean_resizing=False)
        self.model.get_input_embeddings().weight.data = new_input_embeddings
        self.model.get_output_embeddings().weight.data = new_lm_head
        print("Model embedding matrices resized successfully.")

        # --- Stage 2: Model Configuration Update ---
        self._update_model_config(text_boundary_idx, num_new_tokens)
        
        # --- Stage 3: Tokenizer Update ---
        self._update_tokenizer(text_boundary_idx, num_new_tokens)
        
        print("\n--- Embedding Surgery Complete ---")
        return self.model, self.tokenizer
    
    def _create_new_embedding_matrix(self, old_embedding_layer, boundary_idx, num_new, new_size):
        """Helper to create and populate the new, larger embedding matrix."""
        config = self.model.config.text_config
        new_matrix = torch.zeros(new_size, config.hidden_size, device=old_embedding_layer.weight.device, dtype=old_embedding_layer.weight.dtype)
        
        # 1. Copy original text embeddings
        new_matrix[0:boundary_idx] = old_embedding_layer.weight[0:boundary_idx]
        
        # 2. Initialize new token embeddings with the mean
        mean_embeddings = old_embedding_layer.weight[100:20000].mean(dim=0)
        for i in range(num_new):
            new_matrix[boundary_idx + i] = mean_embeddings
            
        # 3. Copy multimodal and special embeddings to their new positions
        new_matrix[boundary_idx + num_new:] = old_embedding_layer.weight[boundary_idx:]
        return new_matrix
    
    def _update_model_config(self, boundary_idx, num_new):
        """Shifts token ID references in the model's configuration."""
        print("Updating model configuration...")
        attrs_to_shift = ['boi_token_id', 'eoi_token_id', 'image_token_id', 'boa_token_id', 'eoa_token_id', 'audio_token_id']
        vision_attrs_to_shift, audio_attrs_to_shift = ['vocab_offset'], ['vocab_offset']

        for attr in attrs_to_shift:
            if getattr(self.model.config, attr) >= boundary_idx:
                setattr(self.model.config, attr, getattr(self.model.config, attr) + num_new)
        for attr in vision_attrs_to_shift:
            if getattr(self.model.config.vision_config, attr) >= boundary_idx:
                setattr(self.model.config.vision_config, attr, getattr(self.model.config.vision_config, attr) + num_new)
        for attr in audio_attrs_to_shift:
            if getattr(self.model.config.audio_config, attr) >= boundary_idx:
                setattr(self.model.config.audio_config, attr, getattr(self.model.config.audio_config, attr) + num_new)
        
        self.model.config.text_config.vocab_size += num_new

    def _update_tokenizer(self, boundary_idx, num_new):
        """Rebuilds the tokenizer with the new vocabulary."""
        print("Updating tokenizer...")
        temp_file = "temp_tokenizer.json"
        self.tokenizer.tokenizer.backend_tokenizer.save(temp_file)

        with open(temp_file, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        original_vocab = tokenizer_data['model']['vocab']
        new_vocab = {}
        
        for token, token_id in original_vocab.items():
            if token_id < boundary_idx:
                new_vocab[token] = token_id
        
        for i, token in enumerate(self.surgery_config.new_special_tokens):
            new_vocab[token] = boundary_idx + i
        
        for token, token_id in original_vocab.items():
            if token_id >= boundary_idx:
                new_vocab[token] = token_id + num_new

        tokenizer_data['model']['vocab'] = new_vocab
        for i, token in enumerate(self.surgery_config.new_special_tokens):
            tokenizer_data['added_tokens'].append({
                "id": boundary_idx + i, "content": token, "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": False
            })

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        new_backend_tokenizer = Tokenizer.from_file(temp_file)
        self.tokenizer.tokenizer = GemmaTokenizerFast(tokenizer_object=new_backend_tokenizer)
        os.remove(temp_file)
    
    def prepare_model_for_dapt(self):
        """Freezes all layers except the token embeddings and lm_head for DAPT."""
        print("--- Preparing model for DAPT (training embeddings and lm_head) ---")
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze input embeddings
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'): # For base models
             self.model.model.embed_tokens.weight.requires_grad = True
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'embed_tokens'): # For Gemma 3N
            self.model.language_model.embed_tokens.weight.requires_grad = True

        # Unfreeze output embeddings (lm_head)
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.weight.requires_grad = True

        print_trainable_parameters(self.model)
        return self.model

    def apply_sft_peft(self, peft_config: SFTPEFTConfig):
        """Applies LoRA adapters to the model for SFT."""
        print("--- Applying PEFT (LoRA) for SFT ---")
        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # 

            
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            target_modules=peft_config.target_modules,
            random_state=peft_config.random_state,
        )
        print_trainable_parameters(self.model)
        return self.model