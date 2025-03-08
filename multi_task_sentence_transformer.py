
#task 2: Multi task learning

#Head A - text classification
#use the same model from the task 1 (Sentence transformer)
#Add a classification head on top
#It is a feed forward layer, input_dim = 768, output_dim= number of classes
#With a softmax activation to obtain the probability of each class

#Head B- NER
#modify the sentence transformer model for NER task.
#Currently, the sentence transformer returns a single vector for a text.
#The model needs to be modified to obtain the vectors for all the tokens.
#Each token vector needs to be classified as an NER tags (BIO tags representation)
#Each Entity class will have B and I tags indicating the beginning and middle of the entity respectively.
#For ex: Donald Trump is the president. Donald: B-PER; Trump: I-PER; is - O

import torch
from transformers import AutoTokenizer, AutoModel


class MultiTaskSentenceTransformer(torch.nn.Module):
    def __init__(self,
                 model_name,
                 num_classifier_labels: int = 1,
                 num_ner_labels: int = 1,
                 non_mean_pooling: bool = False,
                 display_token_embeddings: bool = False):

        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        hidden_dim = self.model.config.hidden_size
        self.classifier_head = torch.nn.Linear(hidden_dim, num_classifier_labels).to(self.device)
        self.ner_head = torch.nn.Linear(hidden_dim, num_ner_labels).to(self.device)

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embeddings

    def non_mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state
        sentence_embeddings = token_embeddings[:, 0] #[CLS] token
        return sentence_embeddings

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids, attention_mask)
        sentence_vector = self.mean_pooling(output, attention_mask) if self.non_mean_pooling else self.non_mean_pooling(output, attention_mask)
        token_vectors = output.last_hidden_state

        classifier_logits = self.classifier_head(sentence_vector)
        ner_logits = self.ner_head(token_vectors)

        return classifier_logits, ner_logits

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
          classifier_logits, ner_logits = self.forward(input_ids, attention_mask)
          return torch.argmax(classifier_logits, dim=-1), torch.argmax(ner_logits, dim=-1)
