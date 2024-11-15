from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import pandas as pd

# 1. Load Pretrained Model and Tokenizer
model_name = "gpt2"  # Replace with any Hugging Face model like EleutherAI/gpt-neo-125M or bigger
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Prepare Dataset
def load_dataset(file_path, tokenizer, block_size=512):
    """
    Loads and tokenizes a text dataset for language modeling.
    """
    dataset = TextDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        block_size=block_size,
    )
    return dataset

# Adjust the sequence length
max_length = 1024

# Truncate or split text into chunks
def split_long_text(text, max_length):
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokenized_text[i:i + max_length] for i in range(0, len(tokenized_text), max_length)]
    return chunks

train_path = "data/train.txt"  
val_path = "data/val.txt"
save_path = "./bedtime_story_model"      

# train = pd.read_csv(train_path)
# train['text'] = train['text'].fillna('').astype(str)

train_dataset = load_dataset(train_path, tokenizer)
val_dataset = load_dataset(val_path, tokenizer)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal Language Modeling (not MLM)
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./finetuneTransformer/output/bedtime_story_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./finetuneTransformer/logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=5e-5,
    save_strategy="steps",
    fp16=True, 
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 7. Save the Model
trainer.save_model(save_path)

model.eval()

