"""

STEP 0 - PRE REQUISITES

"""


# python -m spacy download en_core_web_lg

# Import libraries
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

# nlp = spacy.load("en_core_web_lg")
#
# with open("food.txt") as file:
#     dataset = file.read()
#
# doc = nlp(dataset)
# print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
#
# """
#
# STEP 1 - TRAIN DATA
#
# """
#
# # Prepare training data
#
# words = ["ketchup", "pasta", "carrot", "pizza",
#          "garlic", "tomato sauce", "basil", "carbonara",
#          "eggs", "cheek fat", "pancakes", "parmigiana", "eggplant",
#          "fettucine", "heavy cream", "polenta", "risotto", "espresso",
#          "arrosticini", "spaghetti", "fiorentina steak", "pecorino",
#          "maccherone", "nutella", "amaro", "pistachio", "coca-cola",
#          "wine", "pastiera", "watermelon", "cappuccino", "ice cream",
#          "soup", "lemon", "chocolate", "pineapple"]
#
# train_data = []
#
# with open("food.txt") as file:
#     dataset = file.readlines()
#     for sentence in dataset:
#         print("######")
#         print("sentence: ", sentence)
#         print("######")
#         sentence = sentence.lower()
#         entities = []
#         for word in words:
#             word = word.lower()
#             if word in sentence:
#                 start_index = sentence.index(word)
#                 end_index = len(word) + start_index
#                 print("word: ", word)
#                 print("----------------")
#                 print("start index:", start_index)
#                 print("end index:", end_index)
#                 pos = (start_index, end_index, "FOOD")
#                 entities.append(pos)
#         element = (sentence.rstrip('\n'), {"entities": entities})
#
#         train_data.append(element)
#         print('----------------')
#         print("element:", element)
#
#         # ("this is my sentence", {"entities": [0, 4, "PREP"]})
#         # ("this is my sentence", {"entities": [6, 8, "VERB"]})
#
# """
#
# STEP 2 - UPDATE MODEL
#
# """
#
# ner = nlp.get_pipe("ner")
#
# for _, annotations in train_data:
#     for ent in annotations.get("entities"):
#         ner.add_label(ent[2])
#
# # Train model
#
# pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
# unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
#
# with nlp.disable_pipes(*unaffected_pipes):
#     for iteration in range(30):
#         print("Iteration #", iteration)
#
#         random.shuffle(train_data)
#         losses = {}
#
#         batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
#         for batch in batches:
#             for text, annotations in batch:
#                 doc = nlp.make_doc(text)
#                 example = Example.from_dict(doc, annotations)
#                 nlp.update([example], losses=losses, drop=0.1)
#         print("Losses:", losses)



# Save the model

output_dir = Path("/ner/")
# nlp.to_disk(output_dir)
# print("Saved correctly!")

"""

STEP 3 - TEST THE UPDATED MODEL

"""

print("Loading model...")
nlp_updated = spacy.load(output_dir)

# old sentence, old word
doc = nlp_updated("I don't like pizza with chocolate.")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# old sentence, new word
doc = nlp_updated("in carbonara, parmigiano is not used.")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# new sentence, new word
doc = nlp_updated("In Rimini they don't make piadina icecream")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# new sentence, no word
doc = nlp_updated("Fabio likes full-stack development")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])
