# Named Entity Recognition with Fine-Tuned BERT

This project demonstrates how to fine-tune a pre-trained BERT model (`bert-base-uncased`) for **Named Entity Recognition (NER)** using the **CoNLL-2003** dataset. The goal is to teach BERT how to identify entities like people, organizations, locations, and miscellaneous names in raw text.

---

## What I Did

I walked through the full NER pipeline using the Hugging Face ecosystem:

### 1. **Dataset Loading**
I used the Hugging Face `datasets` library to load the CoNLL-2003 dataset — a benchmark dataset for NER tasks.

### 2. **Tokenization & Label Alignment**
I tokenized the input sentences using BERT’s tokenizer, carefully aligning each word’s label with its subword tokens. Tokens like `"Apple"` and `"##ton"` are treated as part of the same word but still need correct label handling.

### 3. **Model Setup**
Then loaded a pre-trained BERT model (`bert-base-uncased`) and adapted it for token classification by setting the number of output labels to match the NER tags in the dataset.

### 4. **Training**
I fine-tuned the model using Hugging Face’s `Trainer` API. I trained for 3 epochs with early logging, saving checkpoints and evaluating on the validation set during training.

### 5. **Evaluation**
The model was evaluated using the `seqeval` metric (designed for sequence labeling tasks), and I reported key performance metrics: **precision**, **recall**, **F1 score**, and **accuracy**.

### 6. **Inference**
Finally, tested the trained model on a custom sentence:  
```text
"Bill Gates is the Founder of Microsoft"
