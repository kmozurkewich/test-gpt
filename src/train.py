"""
Main training script for genealogical LLM fine-tuning.
"""
import argparse
import torch
from pathlib import Path
from data_prep import prepare_genealogy_data
from model_config import setup_model, create_trainer, save_model, DEFAULT_MAX_SEQ_LENGTH

def parse_args():
    parser = argparse.ArgumentParser(description='Train a genealogy-focused LLM')
    parser.add_argument('--data-file', type=str, required=True,
                      help='Path to genealogy CSV file')
    parser.add_argument('--output-dir', type=str, default='./model_output',
                      help='Directory to save the model')
    parser.add_argument('--batch-size', type=int, default=2,
                      help='Training batch size')
    parser.add_argument('--grad-accum', type=int, default=4,
                      help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5,  # Increased epochs
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,  # Adjusted learning rate
                      help='Learning rate')
    parser.add_argument('--max-seq-length', type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                      help='Maximum sequence length')
    parser.add_argument('--save-gguf', action='store_true',
                      help='Save model in GGUF format')
    return parser.parse_args()

def train_model(args):
    """Execute the training pipeline."""
    
    print("\n=== Genealogical Record Assistant Training ===\n")
    
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model(max_seq_length=args.max_seq_length)
    
    print("\nPreparing genealogy dataset...")
    dataset = prepare_genealogy_data(args.data_file, tokenizer)
    print(f"Generated {len(dataset)} training examples")
    
    # Print sample of training data
    print("\nSample training examples:")
    for idx in range(min(2, len(dataset))):
        print(f"\nExample {idx + 1}:")
        print(dataset[idx]['text'][:500] + "...")
    
    print("\nConfiguring trainer...")
    trainer = create_trainer(
        model,
        tokenizer,
        dataset,
        args.output_dir,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,  # 10% warmup
    )
    
    # Print GPU stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU: {gpu_stats.name}")
        print(f"Total GPU memory: {max_memory} GB")
        print(f"Initial reserved memory: {start_gpu_memory} GB")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    
    print("\nStarting training...")
    try:
        trainer_stats = trainer.train()
        
        # Print training summary
        runtime = trainer_stats.metrics['train_runtime']
        print(f"\nTraining completed in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
        final_loss = trainer_stats.metrics.get('train_loss', 'N/A')
        print(f"Final training loss: {final_loss}")
        
        print("\nSaving model...")
        save_model(
            model,
            tokenizer,
            args.output_dir,
            save_gguf=args.save_gguf
        )
        
        print("\nTraining successfully completed!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

def main():
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save training arguments for reference
    with open(Path(args.output_dir) / "training_args.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Run training pipeline
    train_model(args)

if __name__ == "__main__":
    main()