from nltk.cluster.util import cosine_distance
import numpy as np

def get_cosine_similarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]

    words_all = list(set(sentence1 + sentence2))

    array1 = [0] * len(words_all)
    array2 = [0] * len(words_all)

    for w in sentence1:
        if w in stopwords:
            continue
        array1[words_all.index(w)] += 1

    # build the vector for the second sentence
    for w in sentence2:
        if w in stopwords:
            continue
        array2[words_all.index(w)] += 1

    return 1 - cosine_distance(array1, array2)


def cos_sim_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            matrix[idx1][idx2] = get_cosine_similarity(sentences[idx1], sentences[idx2], stop_words)

    return matrix

sentence1 = "This is a first sentence"
sentence2 = "This is a second sentence"
sentence3 = "I don't know if it's clear, but pineapple on pizza is oltrageous."

print(cos_sim_matrix([sentence1, sentence2, sentence3], []))