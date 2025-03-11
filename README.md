# Genealogy LLM Fine-tuning Tool

A Python tool for fine-tuning LLama-3 models with genealogical data using Unsloth.

## Project Structure
```
genealogy-llm/
├── README.md
├── requirements.txt
├── data/
│   └── history_records.csv
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Data preprocessing
│   ├── model_config.py     # Model configuration
│   ├── train.py           # Training setup and execution
│   └── utils.py           # Helper functions
└── notebooks/
    └── exploration.ipynb  # Data exploration and testing
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your genealogical data CSV in the `data/` directory
2. Run the training script:
```bash
python src/train.py --data-file data/history_records.csv
```

## Dependencies
- unsloth
- torch
- transformers
- pandas
- trl
