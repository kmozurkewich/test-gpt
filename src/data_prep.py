"""
Data preparation module for genealogical LLM fine-tuning with strict relationship handling.
"""
from typing import Dict, List, Optional
import pandas as pd
from datasets import Dataset
from unsloth import standardize_sharegpt, apply_chat_template

class GenealogyDataPrep:
    def __init__(self, csv_path: str):
        """Initialize with path to genealogy CSV file."""
        self.df = pd.read_csv(csv_path)
        self.records = {}  # Store validated records
        self._prepare_records()
        
    def _prepare_records(self) -> None:
        """Prepare validated records for each person."""
        for _, row in self.df.iterrows():
            if pd.notna(row['Name']):
                name = row['Name'].strip()
                record = {
                    'name': name,
                    'godfather': row['godFather'].strip() if pd.notna(row['godFather']) else None,
                    'parent1': row['parent1'].strip() if pd.notna(row['parent1']) else None,
                    'parent2': row['parent2'].strip() if pd.notna(row['parent2']) else None,
                }
                self.records[name] = record

    def _get_relationship_statement(self, name: str, record: Dict, rel_type: str) -> str:
        """Generate precise relationship statements."""
        if rel_type == 'godfather' and record['godfather']:
            return f"The historical record shows that {record['godfather']} is {name}'s godfather."
        
        if rel_type == 'parents':
            parents = []
            if record['parent1']:
                parents.append(record['parent1'])
            if record['parent2']:
                parents.append(record['parent2'])
            
            if parents:
                parent_str = " and ".join(parents)
                return f"The historical record shows that {parent_str} are {name}'s parents."
        
        return f"The historical record does not show {rel_type} for {name}."

    def _generate_examples(self, name: str, record: Dict) -> List[Dict[str, str]]:
        """Generate training examples for each relationship type."""
        examples = []
        
        def add_qa(question: str, answer: str):
            examples.append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

        # Godfather examples
        godfather_statement = self._get_relationship_statement(name, record, 'godfather')
        add_qa(f"Who is recorded as {name}'s godfather?", godfather_statement)
        add_qa(f"Based on the historical record, who is {name}'s godfather?", godfather_statement)
        
        # Parent examples
        parent_statement = self._get_relationship_statement(name, record, 'parents')
        add_qa(f"Who are {name}'s parents in the record?", parent_statement)
        add_qa(f"Based on the historical record, who are {name}'s parents?", parent_statement)
        add_qa(f"Who are recorded as the parents of {name}?", parent_statement)
        
        # Add negative examples to prevent mixing relationships
        if record['godfather']:
            add_qa(
                f"Are {record['godfather']}'s parents shown in the record?",
                f"The historical record does not show parent information for {record['godfather']}."
            )
        
        return examples

    def prepare_dataset(self, tokenizer) -> Dataset:
        """Prepare dataset with relationship-specific examples."""
        all_examples = []
        
        for name, record in self.records.items():
            examples = self._generate_examples(name, record)
            all_examples.extend(examples)
        
        dataset = Dataset.from_list(all_examples)
        dataset = standardize_sharegpt(dataset)
        
        chat_template = """### Input:
{INPUT}

### Response:
{OUTPUT}

### Input:
{INPUT}

### Response:
{OUTPUT}

"""
        
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        
        return dataset

def prepare_genealogy_data(csv_path: str, tokenizer) -> Dataset:
    """Convenience function to prepare genealogy data."""
    prep = GenealogyDataPrep(csv_path)
    return prep.prepare_dataset(tokenizer)