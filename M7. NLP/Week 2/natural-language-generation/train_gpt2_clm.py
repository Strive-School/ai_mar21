from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
import json
from pathlib import Path
import random
import logging
import math
import torch
import os
from sklearn.model_selection import train_test_split

# 0) SETUP VARS AND LOGGING
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SEED = 10
random.seed(SEED)  # for reproducibility

# 1) LOAD THE DATASET
# Select relevant fields from original format and merge them in plain txt file
# TODO: wrap this with a preprocessing function
# collect sentences
with open("dataset/recipes_raw_nosource_ar.json") as fn:
  recipes = json.load(fn)

# TODO: wrap the data collection into a function
dataset_path = Path('dataset')
sentences = []
for file in dataset_path.iterdir():
  if file.suffix == '.json':
     with open(file) as fn:
       recipes = json.load(fn)
     for id in recipes.keys():
         try:
             title = recipes[id]['title']
             ingredients = ', '.join([ing for ing in recipes[id]['ingredients']])
             instructions = recipes[id]['instructions']
             sentence = f"{title}, {ingredients}, {instructions}"
             if sentence != '':
                 sentences.append(sentence)
         except KeyError:
             continue

# clean sentences
# TODO: add further cleaning steps
def clean(sentence):
    sentence = sentence.replace('ADVERTISEMENT', '')  # replace repetetive words
    sentence = sentence.replace('\n', ' ')  # replace new line chars
    sentence = sentence.strip()  # strip leading and trailing white-spaces
    return sentence

sentences = list(map(clean, sentences))  # map method.
# sentences = [clean(sentence) for sentence in sentences]  # list comprehension method
logger.info(f"Total number of sentences: {len(sentences)}")

# Create train and dev splits
# split into train/dev
# TODO: alternatively, we could use the `datasets.Dataset.train_test_split()` method
SEED = 10  # set seed var for reproducibility
train_sentences, test_sentences = train_test_split(sentences,
                                                   test_size=0.1,
                                                   # change the train_size for rapid testing (for example, use 0.1)
                                                   train_size=0.9,
                                                   random_state=SEED)

# write into files
for split, sents in zip(['train', 'test'], [train_sentences, test_sentences]):
    with open(f"{split}.txt", 'w', encoding='utf-8') as fn:
        fn.write('\n'.join(sents))

# Write to file
logger.info(f"Created splits of size,"
            f"train.json: {len(train_sentences)}, "
            f"test.json: {len(test_sentences)}")

# Use the "load_dataset" method with the "json" builder to create the features
dataset = load_dataset('text', data_files={'train': 'train.txt', 'test': 'test.txt'})

# 2) TOKENIZE DATA AND PREPARE INPUTS AND LABELS
# Instantiate tokenizer
from transformers import AutoTokenizer
pretrained_model = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)

# Fefine a function to tokenize the dataset and return the text indices.
# We also add trailing <|endoftext|> special token
def tokenize_sentence(dataset):
    # As we can see, there is no padding since the PAD token is not originally used by GPT-2.
    # We could perform padding by adding the PAD token to the vocabulary with the method `add_special_tokens()`
    return tokenizer([f"{sentence} {tokenizer.eos_token}" for sentence in dataset['text']])

# apply to dataset object
dataset_features = dataset.map(tokenize_sentence,
                               batched=True,
                               remove_columns=['text'],
                               desc='Tokenizing train and test splits')

# Taken from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Group sentences in batches of equal size to avoid padding
# We use an adaptation of the `group_text` function for that purpose
def group_texts(examples):
    # Concatenate all texts.
    block_size = 512  # set the "blocks" to half of the maximum GPT-2 model length (1024)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # # Add labels to the dataset_features
    # # Since the task is language modelling, the labels to predict are actually the input indices "shifted"

    # result["labels"] = result["input_ids"].copy()
    return result

# apply the group function to the dataset
dataset_grouped = dataset_features.map(group_texts,
                                       batched=True,
                                       desc='Group sentences in blocks of equal size (512)')

# Add "labels" column to the dataset_features.
# To modify the dataset structure, we use the `dataset.map()` method
def add_labels(dataset):
    # Since the task is language modelling, the labels to predict are actually
    # the input indices shifted forward by one element (token)
    dataset['labels'] = dataset['input_ids'].copy()
    return dataset

dataset_for_lm = dataset_grouped.map(add_labels,
                                     batched=True,
                                     desc='Add labels to create data for language model training')


# 3) TRAIN THE CAUSAL LANGUAGE MODEL
# We use the "Trainer" API to perform the training loop
# TODO: experiment with different model configuration and batch sizes until
# the models fits into GPU memory (otherwise it generated CUDA-out-of-memory error)
# The model is instantiated from the pretrained GPT-2 model
# Here, I reduced the number of attention head and layers,
# to significantly reduce the model size and make sure it fits in the GPU memory
config = AutoConfig.from_pretrained(pretrained_model)
model = AutoModelForCausalLM.from_pretrained(pretrained_model,
                                             config=config)

# Again, we simulate a batch size of 8 by settint the `gradient_accumulation_steps` parameters
no_cuda = not bool(torch.cuda.is_available())

if no_cuda:
  print(f"Training on CPUs")
else:
  print(f"Training on GPU")

training_args = TrainingArguments(no_cuda=torch.cuda.is_available(),
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=8, # virtually incremente the batch_size by a factor of 8
                                  evaluation_strategy='epoch',
                                  save_strategy='epoch',
                                  logging_steps=100,
                                  logging_dir='gpt2-full-bs-32-recipes/tb',  # where to store the tensorboard
                                  num_train_epochs=3,
                                  output_dir='gpt2-full-bs-32-recipes')

# Start the training!
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_for_lm['train'],
    eval_dataset=dataset_for_lm['test'], # we use the test set as validation set
    tokenizer=tokenizer,
    # Data collator is used to create batches from data.
    # When a tokenizer is passed the default to DataCollatorWithPadding is used.
    # So we change it since our model do not use PAD tokens
    data_collator=default_data_collator,
)

logger.info('Training...')
train_results = trainer.train()
trainer.save_model('gpt2-recipes')

# Save the metrics (loss on the training data in our case)
metrics_train = train_results.metrics
trainer.log_metrics("train", metrics_train)
trainer.save_metrics("train", metrics_train)
trainer.save_state()

# 4) EVALUATE ON DEV SET WITH PERPLEXITY
metrics_eval = trainer.evaluate()

# compute perplexity
try:
    perplexity = math.exp(metrics_eval["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics_eval["perplexity"] = perplexity

# Save the metrics (loss on the training data in our case)
trainer.log_metrics("eval", metrics_eval)
trainer.save_metrics("eval", metrics_eval)
