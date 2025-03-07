
#Dataset creation
#adding NER tags to ag-news dataset
#dataset will then have labels for classifier and NER


import spacy
from datasets import load_dataset, DatasetDict
nlp = spacy.load("en_core_web_sm")
NOT_REQUIRED_ENT_TYPES = ['CARDINAL', 'DATE', 'QUANTITY', 'ORDINAL', 'TIME']

def get_ner_tags(text):
    doc = nlp(text)
    ner_tags = []
    j = 0
    for token in doc:     
        if token.ent_type_ not in NOT_REQUIRED_ENT_TYPES:
          
          if token.ent_type_ != 'O':        
            ner_tags.append((token.text, token.ent_iob_ + "-" + token.ent_type_))
          else:
            ner_tags.append((token.text, token.ent_iob_))
        else:
          ner_tags.append((token.text, "O"))
        
    return ner_tags


def process_dataset(dataset):
    processed_dataset = dataset.map(
        lambda example: {"ner_tags": get_ner_tags(example["text"])},
        batched=False
    )
    processed_dataset = processed_dataset.map(
        lambda example: {"words": [token[0] for token in example["ner_tags"]]},
        batched=False
    )
    processed_dataset = processed_dataset.map(
        lambda example: {"ner_tags": [token[1] for token in example["ner_tags"]]},
        batched=False
    )
    return processed_dataset

def create_dataset():
  dataset = load_dataset("ag_news")

  train_dataset = dataset["train"].select(range(1000))
  test_dataset = dataset["test"].select(range(200))
  dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

  train_dataset = process_dataset(train_dataset)
  test_dataset = process_dataset(test_dataset)
  dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
  return dataset
