
from create_ner_dataset import create_dataset
from sentenceTransformer import SentenceTransformer
from multi_task_sentence_transformer import MultiTaskSentenceTransformer
from trainer_sentence_transformer import SentenceTransformerTrainer


dataset, tag_set = create_dataset()
classifier_labels = dataset.column_names['train'][0] == 'label'
model = MultiTaskSentenceTransformer(model_name='bert-base-uncased', num_classifier_labels= len(classifier_labels), num_ner_labels=len(list(tag_set)))
trainer = SentenceTransformerTrainer(model, dataset, tag_set)
trainer.run()
