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
from tqdm import tqdm

class SentenceTransformerTrainer():
  def __init__(self,
               model,
               dataset,
               tag_set,
               freeze_transformer: bool = False,
               freeze_classifier: bool = False,
               freeze_ner: bool = False):


    self.model = model
    self.dataset = dataset 
    self.label_map = {label: i for i, label in enumerate(tag_set)}

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
        classifier=True if 'label' in self.dataset.column_names["train"] else False,
        ner=True if 'ner_tags' in self.dataset.column_names["train"] else False
    )

    # Separate loss function for each head
    


  def tokenize_classifier_data(self, batch):
    return self.tokenizer(batch["text"], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

  def tokenize_ner_data(self, batch):
    tokenized_inputs = self.tokenizer(batch["words"], padding='max_length', is_split_into_words = True,   truncation=True, max_length=128, return_tensors='pt')
    #the words key has the list of words for each sentence

    # aligning tags and words
    # adding -100 for 'O' Tags as they are irrelevant
    # -100 allows the loss function to ignore the O tags
    
    # Create a label mapping if it doesn't exist
    

    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Map the label to its numerical ID
                label_ids.append(self.label_map.get(label[word_idx], -100))  
            else:
                label_ids.append(-100)
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
        torch.tensor(tokenized_dataset["train"]["ner_labels"])
    )
    test = TensorDataset(
        torch.tensor(tokenized_dataset["test"]["input_ids"]),
        torch.tensor(tokenized_dataset["test"]["attention_mask"]),
        torch.tensor(tokenized_dataset["test"]["label"]),
        torch.tensor(tokenized_dataset["test"]["ner_labels"])
    )
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)
    return train_loader, test_loader

  def run(self, epochs=3):
    classifier_loss_function = torch.nn.CrossEntropyLoss()
    ner_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
    
    for epoch in tqdm(range(epochs)):
        
        self.model.train()
        for batch in self.train_loader:
            input_ids, attention_mask, labels, ner_labels = batch
            input_ids, attention_mask, labels, ner_labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device), ner_labels.to(self.device)

            optimizer.zero_grad()

            classifier_logits, ner_logits = self.model(input_ids, attention_mask)
            


            classifier_loss = classifier_loss_function(classifier_logits, labels)
            
            ner_loss = ner_loss_function(ner_logits.view(-1, ner_logits.shape[-1]), ner_labels.view(-1))
            

            #cumalative loss in case NER labels are present
            loss = classifier_loss + ner_loss

            loss.backward()
            optimizer.step()
            
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

  def evaluate(self):
    self.model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
      total_correct = {
          "classifier": 0,
          "ner": 0
      }
      total_samples = {
          "classifier": 0,
          "ner": 0
      }
      for batch in self.test_loader:
        input_ids, attention_mask, labels, ner_labels = batch
        input_ids, attention_mask, labels, ner_labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device) , ner_labels.to(self.device)
        
        #classifier labels
        classifier_label_pred, ner_label_pred = self.model.predict(input_ids, attention_mask)

        
        
        total_correct["classifier"] += (classifier_label_pred == labels).sum().item()
        total_samples["classifier"] += labels.size(0)
        total_correct["ner"] += (ner_label_pred == ner_labels).sum().item()
        total_samples["ner"] += ner_labels.size(0)
        
    accuracy = {
        "classifier": total_correct["classifier"] / total_samples["classifier"],
        "ner": total_correct["ner"] / total_samples["ner"]
    }
    print(f"Test Accuracy: {accuracy}")
