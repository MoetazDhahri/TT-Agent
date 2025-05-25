from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import numpy as np
import psutil
import os

# Function to check system memory
def check_memory():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    print(f"Total RAM: {total_gb:.2f} GB, Available: {available_gb:.2f} GB")
    if available_gb < 2:
        print("Warning: Low memory. Training may crash.")
        return False
    return True

# Function to compute perplexity
def compute_perplexity(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels))
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}

# Function to test the chatbot with sample questions
def test_chatbot(model, tokenizer, questions, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)
    
    for question in questions:
        input_text = f"User: {question} Assistant:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Question: {question}")
        print(f"Response: {response}")
        print("-" * 50)

# Main execution
def main():
    try:
        # Check memory before starting
        if not check_memory():
            print("Insufficient memory. Consider closing other applications or using a smaller model.")
            return
        
        # Load preprocessed datasets (Colab-compatible paths)
        train_dataset = load_from_disk("/content/train_chatbot_dataset")
        val_dataset = load_from_disk("/content/val_chatbot_dataset")
        
        # Debug: Inspect dataset
        print("Sample training data:", train_dataset[0])
        
        # Load model and tokenizer
        model_name = "google/flan-t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Ensure inputs are strings
            inputs = [f"User: {q} Assistant:" if isinstance(q, str) else "" for q in examples["question"]]
            targets = [a if isinstance(a, str) else "" for a in examples["answer"]]
            
            # Tokenize inputs
            model_inputs = tokenizer(
                inputs,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize targets (labels)
            model_targets = tokenizer(
                targets,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Debug: Check tokenized output
            print("Sample input_ids:", model_inputs["input_ids"][0][:10])
            print("Sample labels:", model_targets["input_ids"][0][:10])
            
            # Replace padding token id in labels with -100 to ignore in loss
            labels = model_targets["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels
            }
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer"])
        
        # Debug: Inspect tokenized dataset
        print("Sample tokenized data:", train_dataset[0])
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="/content/chatbot_model",
            eval_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="perplexity",
            greater_is_better=False,
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),  # Enable for GPU
            report_to="none"
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_perplexity
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        print("Validation Results:", eval_results)
        
        # Save the model and tokenizer
        trainer.save_model("/content/chatbot_model")
        tokenizer.save_pretrained("/content/chatbot_model")
        
        # Test with sample questions
        sample_questions = [
            "Quel est le tarif d'appel pour l'offre 'Trankil' ?",
            "Où se trouve l'agence Tunisie Telecom Kheireddine Pacha ?",
            "Quels sont les détails du forfait 'El 3échra' de 4 heures ?",
            "Comment activer Internet صبّة ?"
        ]
        
        print("\nTesting chatbot with sample questions:")
        test_chatbot(model, tokenizer, sample_questions)
    
    except RuntimeError as e:
        print(f"Training failed due to resource error: {str(e)}")
        print("Try closing other applications or reducing model size.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()