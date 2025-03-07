
from create_ner_dataset import create_dataset
from sentenceTransformer import SentenceTransformer
from multi_task_sentence_transformer import MultiTaskSentenceTransformer
from trainer_sentence_transformer import SentenceTransformerTrainer


dataset = create_dataset()
classifier_labels = dataset.column_names['train'][0] == 'label'
ner_labels = dataset.column_names['train'][0] == 'ner_tags'
model = MultiTaskSentenceTransformer(model_name='bert-base-uncased', num_classifier_labels= len(classifier_labels), num_ner_labels=len(ner_labels))
trainer = SentenceTransformerTrainer(model, dataset)
trainer.run()
