
"""
Before running the script, 
ensure that you have your dataset ready in a format suitable for LLAMA2 
(typically a JSON file with 'question' and 'answer' fields).
"""

# bash
"""
python -m venv llama2-env
source llama2-env/bin/activate
pip install transformers datasets torch
"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the tokenizer and model
model_name = 'allenai/llama2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    qa_pairs = [f'Question: {q} Answer: {a}' for q, a in zip(questions, answers)]
    return qa_pairs

qa_pairs = load_dataset('path_to_your_dataset.json')
with open('finetuned_corpus.txt', 'w') as f:
    for pair in qa_pairs:
        f.write(f"{pair}\n")

# Prepare dataset for training
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='finetuned_corpus.txt',
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training settings
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    overwrite_output_dir=True,       # overwrite the content of the output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=4,    # batch size for evaluation
    eval_steps=400,                  # Number of update steps between two evaluations.
    save_steps=800,                  # after # steps model is saved
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('path_to_save_finetuned_model')
