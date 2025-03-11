"""
Model configuration module for genealogical LLM fine-tuning.
"""
from typing import Optional, Tuple
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer

DEFAULT_MAX_SEQ_LENGTH = 2048

def setup_model(
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
    """Set up the model and tokenizer for fine-tuning."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    
    # Configure LoRA with more focused parameters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
    )
    
    return model, tokenizer

def create_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    batch_size: int = 2,
    grad_accumulation: int = 4,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 5,
    warmup_ratio: float = 0.1,
) -> SFTTrainer:
    """Create and configure the trainer with stricter parameters."""
    
    # Calculate steps
    total_num_examples = len(dataset)
    effective_batch_size = batch_size * grad_accumulation
    steps_per_epoch = total_num_examples // effective_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_steps=100,
        seed=42,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        group_by_length=True,
        remove_unused_columns=True,  # Changed to True
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    return trainer

def save_model(
    model: FastLanguageModel,
    tokenizer: PreTrainedTokenizer,
    save_path: str,
    save_gguf: bool = True,
    quantization: str = "q8_0"
) -> None:
    """Save the fine-tuned model."""
    print("Saving model weights...")
    model.save_pretrained(save_path)
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained(save_path)
    
    if save_gguf:
        print("Converting to GGUF format...")
        model.save_pretrained_gguf(
            save_path,
            tokenizer,
            quantization_method=quantization
        )