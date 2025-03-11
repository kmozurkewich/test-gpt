"""
Inference script for the fine-tuned genealogy model.
"""
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from utils import run_inference, format_genealogy_query

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on genealogy model')
    parser.add_argument('--model-dir', type=str, required=True,
                      help='Directory containing the fine-tuned model')
    parser.add_argument('--name', type=str,
                      help='Name of person to query about')
    parser.add_argument('--relationship-type', type=str, 
                      choices=['godfather', 'godmother', 'parents'],
                      help='Type of relationship to query about')
    parser.add_argument('--query-type', type=str, default='general',
                      choices=['general', 'specific_relation', 'birth_info'],
                      help='Type of query to make')
    parser.add_argument('--max-tokens', type=int, default=128,
                      help='Maximum tokens to generate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading model...")
    # Load the fine-tuned model with PEFT
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    if args.relationship_type:
        query = f"Based on the historical record, who is recorded as {args.name}'s {args.relationship_type}?"
    else:
        query = f"Based on the historical record, please tell me all recorded information about {args.name}."
    
    print("\nQuery:", query)
    print("\nGenerating response...")
    response = run_inference(
        model,
        tokenizer,
        query,
        max_new_tokens=args.max_tokens
    )
    print("\nResponse:", response)

if __name__ == "__main__":
    main()