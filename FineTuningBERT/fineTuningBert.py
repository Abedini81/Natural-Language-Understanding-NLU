import json
import numpy as np
import evaluate
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset

# Load dataset (renamed to 'dataset_conll' for uniqueness)
dataset_conll = load_dataset("conll2003", trust_remote_code=True)

# Load tokenizer (changed the variable name to 'bert_tokenizer')
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function (modified for clarity and variable renaming)
def custom_tokenizer_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = bert_tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    all_labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Tokenize datasets (changed variable name to 'tokenized_conll')
tokenized_conll = dataset_conll.map(custom_tokenizer_and_align_labels, batched=True)

# Data collator (renamed to 'collator')
collator = DataCollatorForTokenClassification(tokenizer=bert_tokenizer)

# Load model (changed variable name to 'ner_model')
label_list = dataset_conll["train"].features["ner_tags"].feature.names
ner_model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_list)
)

# Create id2label and label2id mappings (renamed variables)
id2label_mapping = {i: label for i, label in enumerate(label_list)}
label2id_mapping = {label: i for i, label in enumerate(label_list)}
ner_model.config.id2label = id2label_mapping
ner_model.config.label2id = label2id_mapping

# Define compute metrics function (renamed and refactored)
metric = evaluate.load("seqeval")

def calculate_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = pred_logits.argmax(axis=2)

    # Remove ignored index (-100) and map predictions/labels to their true values
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training arguments (renamed some variables)
training_args = TrainingArguments(
    "ner-experiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./training_logs",
    logging_strategy="epoch",
)

# Initialize Trainer (renamed 'trainer_instance')
trainer_instance = Trainer(
    model=ner_model,
    args=training_args,
    train_dataset=tokenized_conll["train"],
    eval_dataset=tokenized_conll["validation"],
    data_collator=collator,
    tokenizer=bert_tokenizer,
    compute_metrics=calculate_metrics,
)

# Train the model
trainer_instance.train()

# Save the model and tokenizer (renamed paths to "ner_model_final")
ner_model.save_pretrained("ner_model_final")
bert_tokenizer.save_pretrained("ner_model_final")

# Evaluate the model (renamed 'trainer_instance.evaluate()' for consistency)
evaluation_results = trainer_instance.evaluate()
print("Evaluation Results:", evaluation_results)

# Example inference using pipeline (slightly modified print statements)
from transformers import pipeline
nlp_pipeline = pipeline("ner", model=ner_model.to("cpu"), tokenizer=bert_tokenizer)
example_text = "Bill Gates is the Founder of Microsoft"
ner_output = nlp_pipeline(example_text)
print("Named Entity Recognition Results:", ner_output)
