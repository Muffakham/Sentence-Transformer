

#Created a trainer class for training the multi task transformer model
#datasets of two types are created one for classification and the other for NER
#loss for both is calculated and added for a cumalative loss
#both the heads and the transformer parameters are updated.
#available options to freeze the parameters of trnasformer or the either head.


import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SentenceTransformerTrainer():
  def __init__(self, 
               model, 
               dataset,
               freeze_transformer: bool = False,
               freeze_classifier: bool = False,
               freeze_ner: bool = False):
    
    
    self.model = model
    self.dataset = dataset
    
    #freezing layers
    if freeze_transformer:
      for param in self.model.model.parameters():
        param.requires_grad = False
    if freeze_classifier:
      for param in self.model.classifier_head.parameters():
        param.requires_grad = False
    if freeze_ner:
      for param in self.model.ner_head.parameters():
        param.requires_grad = False

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.tokenizer = self.model.tokenizer
    

    #Depending upon the label type
    #if both types of labels are present
    #both the heads will be trained

    self.train_loader, self.test_loader = self.pre_process_data(
        classifier=True if self.dataset.column_names['train'][0] == 'label' else False,
        ner=True if self.dataset.column_names['train'][0] == 'ner_tags' else False
    )

    # Separate loss function for each head
    self.classifier_loss_function = torch.nn.CrossEntropyLoss()
    self.ner_loss_function = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)


  def tokenize_classifier_data(self, batch):
    return self.tokenizer(batch["text"], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
  
  def tokenize_ner_data(self, batch):
    tokenized_inputs = self.tokenizer(batch["words"], padding='max_length', is_split_into_words = True,   truncation=True, max_length=128, return_tensors='pt')
    #the words key has the list of words for each sentence

    # aligning tags and words
    # adding -100 for 'O' Tags as they are irrelevant
    # -100 allows the loss function to ignore the O tags

    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            if word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_list[label[word_idx]] != "O" else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["ner_labels"] = labels
    return tokenized_inputs

  def pre_process_data(self, classifier: bool = False, ner: bool = False):
    
    #creating loaders for train and test data
    if ner:
      tokenized_dataset = self.dataset.map(self.tokenize_ner_data, batched=True)
    else:
      tokenized_dataset = self.dataset.map(self.tokenize_classifier_data, batched=True)

    train = TensorDataset(
        torch.tensor(tokenized_dataset["train"]["input_ids"]),
        torch.tensor(tokenized_dataset["train"]["attention_mask"]),
        torch.tensor(tokenized_dataset["train"]["label"]),
        torch.tensor(tokenized_dataset["train"]["ner_labels"]) if ner else None
    )
    test = TensorDataset(
        torch.tensor(tokenized_dataset["test"]["input_ids"]),
        torch.tensor(tokenized_dataset["test"]["attention_mask"]),
        torch.tensor(tokenized_dataset["test"]["label"]),
        torch.tensor(tokenized_dataset["test"]["ner_labels"]) if ner else None
    )
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)
    return train_loader, test_loader

  def run(self, epochs=3):
    for epoch in range(epochs):
        for batch in self.train_loader:
            input_ids, attention_mask, labels, ner_labels = batch
            input_ids, attention_mask, labels, ner_labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), ner_labels.to(self.device) if ner_labels is not None else None
            classifier_logits, ner_logits = self.model(input_ids, attention_mask)

            
            classifier_loss = self.classifier_loss_function(classifier_logits, labels)
            if ner_labels:
                ner_loss = self.ner_loss_function(ner_logits.view(-1, ner_logits.shape[-1]), labels.view(-1))
                loss = ner_loss

            #cumalative loss in case NER labels are present
            loss = classifier_loss + ner_loss if ner_labels else classifier_loss
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
