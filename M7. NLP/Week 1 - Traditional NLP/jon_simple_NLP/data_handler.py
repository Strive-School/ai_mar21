import numpy as np

text_dir ="./texts/"

def preporcess(pth):
    
    with open(pth,'r') as f:
        
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("?","").replace(",","").replace(".","").replace("\"","").replace("\'","").replace(";","").replace(":","").replace("&","").replace("!","").replace("-","").replace("_","")
        with open(text_dir+"processed.txt",'w+') as f:
            f.writelines(lines)
                
def filter_list(sentences):
    
    i = 0
    while i < len(sentences):
        
        if sentences[i] == []:
            sentences.pop(i)
        else:
            i += 1
            
    return sentences
    
def get_sentences(pth):
    f = open(pth,'r')
    lines = f.readlines()
    sentences = [line.split() for line in lines]
    
    return sentences
    
def get_dics(sentences):
    
    vocabulary = []
    for sentence in sentences:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    
    return word2idx, idx2word, len(vocabulary)

def get_pair(sentences, word2idx):
    pairs = []
    for sentence in sentences:
        indices = [word2idx[word] for word in sentence]
        
        for center_word_pos in range(len(indices)):
            for word in range(-4, 4 + 1): 
                   
                context_word_pos = center_word_pos + word
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                
                context_word_idx = indices[context_word_pos]
                pairs.append((indices[center_word_pos], context_word_idx))

    return np.array(pairs)


def get_dataset():
    
    sentences = get_sentences(text_dir+"processed.txt")
    sentences = filter_list(sentences)
    
    word2idx, _, vocab_size = get_dics(sentences)
    data = get_pair(sentences, word2idx)
    
    return data, vocab_size