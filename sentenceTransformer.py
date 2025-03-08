#Task 1
#Sentence transformer architecture
#the base model is bert-base-uncased (Encoder)
#to obtain sentence embeddings, two approaches can be used
#first is to mean pool embeddings of the individual tokens in a sentence
#second is to use the [CLS] token as it stores the information of the entire sentence


import torch
from transformers import AutoTokenizer, AutoModel


class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name_or_path = 'bert-base-uncased', non_mean_pooling: bool = False, display_token_embeddings: bool = False, train_model: bool = False):

        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.display_token_embeddings = display_token_embeddings
        self.non_mean_pooling = non_mean_pooling
        self.train_model = train_model

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take the average of all tokens in a sequence to get a single vector.
        param model_output: Last hidden states of the model,
        param attention_mask: Attention mask of the input sequence,
        return: Sentence embeddings (single vector)
        """
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def non_mean_pooling(self, model_output, attention_mask):
        """
        Non-Mean Pooling - Take the last hidden state of the first token ([CLS]) in the sequence to get a single vector.
        param model_output: Last hidden states of the model,
        param attention_mask: Attention mask of the input sequence,
        return: Sentence embeddings
        """
        token_embeddings = model_output.last_hidden_state

        sentence_embeddings = token_embeddings[:, 0]

        return sentence_embeddings

    def forward(self, input_ids, attention_mask):

        model_output = self.model(input_ids, attention_mask)

        if self.display_token_embeddings:
            print("embeddings of individual tokens in the sentence/text/sequence")
            print(model_output.last_hidden_state[0])
            print('size of the embeddings before transformation')
            print(model_output.last_hidden_state[0].size())

        if self.non_mean_pooling:
            sentence_embeddings = self.non_mean_pooling(model_output, attention_mask)
        else:
          sentence_embeddings = self.mean_pooling(model_output, attention_mask)

        if self.display_token_embeddings:
            print("sentence embeddings for the sentence/text/sequence")
            print(sentence_embeddings)
            print('size of the embeddings after transformation')
            print(sentence_embeddings.size())

        return sentence_embeddings

    def encode_text(self, text):
        tokenized_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        sentence_embeddings = self.forward(tokenized_input['input_ids'], tokenized_input['attention_mask'])
        return sentence_embeddings
