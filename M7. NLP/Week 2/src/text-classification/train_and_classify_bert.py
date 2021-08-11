# -*- coding: utf-8 -*-
# Import libraries
from datasets import load_dataset
import datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np


## 1) Data preprocessing and loading
# Download a dataset from the HF datasets hub
ag_dataset = load_dataset('ag_news')

# Create a train/dev/test splits
ag_dev_dataset = load_dataset('ag_news', split='train[10%:11%]')
ag_train_dataset = load_dataset('ag_news', split='train[:10%]')
ag_test_dataset = load_dataset('ag_news', split='test[11%:12%]')

# merge the splits in a single `datasets.DatasetDict` object
ag_split = {split: data for split, data in zip(['train', 'test', 'dev'], [ag_train_dataset, ag_test_dataset, ag_dev_dataset])}
ag_dataset_split =  datasets.DatasetDict(ag_split)

# Count the number of labels.
# Important: use all the splits to compute the labels. 
num_labels = len(set(ag_dataset_split['train'].unique('label') + 
                     ag_dataset_split['test'].unique('label') +
                     ag_dataset_split['dev'].unique('label')))

## 2) Prepare the features: tokenizing, padding and truncate
# Define a tokenizer

model_pretrained = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_pretrained)

"""🤔 **Understanding BERT tokenizer**"""

# Define a tokenization function for the dataset used a standard for text classification
def tokenize(dataset):
    sentences = dataset['text']
    return tokenizer(sentences, 
                     padding='max_length',
                     truncation=True)
    
# apply it 
ag_dataset_tokenized = ag_dataset_split.map(tokenize,
                                            batched=True,
                                            remove_columns=['text'],
                                            desc='Tokenize data')

## 3) Train the model

# To perform the classification task, we need to create a custom config 
# by adding the number of labels to predict.
# The number of labels will be used  for computing the sequence classification loss
# Note how this differs from the GPT-2 causal language model training
config = AutoConfig.from_pretrained(model_pretrained,
                                    num_labels=num_labels)

# Instantiate the classification layer
model = AutoModelForSequenceClassification.from_pretrained(model_pretrained,
                                                           config=config)

# Create training arguments and trainer objects
training_args = TrainingArguments(output_dir='bert-ag-news-classification',
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  num_train_epochs=2,
                                  logging_steps=100,
                                  logging_dir='bert-ag-news-classification/tb',
                                  evaluation_strategy='epoch',
                                  no_cuda=not torch.cuda.is_available()
                                  )

trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=training_args,
                  train_dataset=ag_dataset_tokenized['train'],
                  eval_dataset=ag_dataset_tokenized['dev'])

train_result = trainer.train()

# Save model 
trainer.save_model()  

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

## 4) Evaluate the model:
# Run again evaluation on the last step to get the metrics and save them
eval_metrics = trainer.evaluate()

trainer.log_metrics("dev", eval_metrics)
trainer.save_metrics("dev", eval_metrics)
trainer.save_state()

# Use the `trainer.predict()` method
test_result = trainer.predict(test_dataset=ag_dataset_tokenized['test'])
# convert predicted logits to label classes.
predicted_label_ids = np.argmax(test_result.predictions, axis=1)

# map the label ids to the classes using the `int2str` methods of the `datasets.ClassLabel`
predicted_label_names = ag_dataset_tokenized['test'].features['label'].int2str(predicted_label_ids)
print(predicted_label_names)