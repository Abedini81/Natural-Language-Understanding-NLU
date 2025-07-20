# Named Entity Recognition with BERT (Fine-Tuning)

This project is a practical implementation of **Named Entity Recognition (NER)** using a fine-tuned BERT model. I trained the model to detect entities such as people, organizations, locations, and miscellaneous names using the **CoNLL-2003** dataset.

---

## What I Did

### 1. Loaded the Dataset
I used the Hugging Face `datasets` library to load the `conll2003` dataset, which contains token-level NER annotations.

### 2. Tokenized Input and Aligned Labels
I used the BERT tokenizer (`bert-base-uncased`) to tokenize the input text. Since BERT splits words into subwords, I aligned the original NER labels to match the tokenized format and ignored special/padding tokens using `-100`.

### 3. Prepared the Model
Then loaded a pre-trained BERT model for token classification and configured it with the correct number of labels. I also set up mappings between label IDs and label names.

### 4. Fine-Tuned the Model
Using Hugging Face’s `Trainer` and `TrainingArguments`, I fine-tuned the model on the training set for 3 epochs with evaluation and saving done at each epoch.

### 5. Evaluated the Model
I evaluated the model using the `seqeval` metric to compute **precision**, **recall**, **F1 score**, and **accuracy** on the validation set.

### 6. Ran Inference
Finally, I used the trained model in a `pipeline` to perform NER on a custom input sentence:  
```text
"Bill Gates is the Founder of Microsoft"
