# Sentence-Transformer

## Codebase Overview

This project implements a multi-task sentence transformer using the Hugging Face Transformers library. The model is designed to perform both **text classification** and **Named Entity Recognition (NER)**, leveraging a shared BERT-based encoder.

### Key Components

- **sentenceTransformer.py**: Implements a basic sentence transformer using BERT. Provides pooling methods for sentence embeddings and can display token embeddings.
- **multi_task_sentence_transformer.py**: Extends the basic model for multi-task learning, adding two heads: one for text classification and one for NER (BIO tagging).
- **trainer_sentence_transformer.py**: Contains a trainer class for data preprocessing, tokenization, training, evaluation, and supports freezing parts of the model.
- **create_ner_dataset.py**: Prepares the dataset by adding NER tags to AG News using spaCy and aligns NER tags with tokens.
- **train.py**: The main script to run the project, handling dataset creation, model initialization, training, and evaluation.
- **requirements.txt**: Lists dependencies: torch, transformers, spacy, datasets, seqeval.
- **sentence_transformer.ipynb**: (Optional) Jupyter notebook for experimentation or step-by-step development.

### Workflow

1. **Dataset Creation**: `create_ner_dataset.py` processes AG News, adds NER tags, and prepares the data.
2. **Model Initialization**: `multi_task_sentence_transformer.py` defines a model with shared BERT encoder and two task-specific heads.
3. **Training**: `trainer_sentence_transformer.py` handles batching, loss calculation, freezing options, and optimization.
4. **Execution**: `train.py` ties everything together for end-to-end training and evaluation.

### Freezing Parameters
- The trainer supports freezing:
  - The entire network (no training).
  - Only the transformer (train only heads).
  - Only one of the heads (focus on a specific task).

### Dependencies
- Hugging Face Transformers, PyTorch, spaCy, Datasets, Seqeval.

### Entry Point
- Run `train.py` to execute the full pipeline: dataset creation, model training, and evaluation.

## Multi-task Sentence Transformer

This project implements a multi-task sentence transformer model using the Hugging Face Transformers library. The model performs two tasks: text classification and Named Entity Recognition (NER).

## Project Structure

The project is organized as follows:

*   **`sentenceTransformer.py`**: This file implements the basic sentence transformer model. It uses the BERT-base-uncased model as the encoder and provides two pooling methods (mean pooling and [CLS] token pooling) to obtain sentence embeddings. It allows for displaying token embeddings and training the model.
*   **`multi_task_sentence_transformer.py`**: This file extends the sentence transformer for multi-task learning. It adds two heads on top of the base model: a classification head for text classification and an NER head for NER. It modifies the sentence transformer to obtain vectors for all tokens for NER and uses BIO tags representation for entity recognition.
*   **`trainer_sentence_transformer.py`**: This file provides a trainer class for training the multi-task model. It handles dataset preprocessing, tokenization, and training loop. It supports freezing specific layers of the model (transformer, classifier head, or NER head) during training. It uses separate loss functions for each head and combines them for cumulative loss.
*   **`create_ner_dataset.py`**: This file is responsible for creating the NER dataset. It adds NER tags to the AG News dataset using spaCy's named entity recognition capabilities. It preprocesses the dataset, tokenizes the text, and aligns the NER tags with the tokens.
*   **`main.py`**: This is the main script to run the project. It imports necessary modules, creates the dataset, initializes the multi-task sentence transformer model, sets up the trainer, and starts the training process.
*   **`requirements.txt`**: This file lists the project dependencies, including torch, transformers, datasets, numpy, and spacy. These dependencies need to be installed to run the project.

## Frozen paramters training

* **`freezing entire network`**: if the entire network is frozen, no training will happen.
* **`freezing transformer network`**: if the transformer block is frozen, then only the heads will be trained. In this case, the pre-trained knowledge is retained. The transformer block does not learn anyhting. It might be suitable for training task specific heads with in the same domain. if the domain is different, then it is important for the transformer block to be unfrozen, in order to understand domain specific vocab.
* **`freezing task specific head`**: in this case either of the task specific head can be frozen. Therefore only one task will be learnt. Allows for improving task specific performance. In this example, the classification task performs better than the NER task. In order to improve NER, classifier can be frozen and the NER traineed. If there is more data for a specific task, it is benefecial to freeze the other task head. 

## Requirements

To run this project, you need to have the following libraries installed:

*   torch
*   transformers
*   datasets
*   numpy
*   spacy

You can install them using pip install -r requirements.txt
