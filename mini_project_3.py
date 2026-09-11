import numpy as np
from sentence_transformers import SentenceTransformer


def get_word_embedding(word, model): #embedding trả về 1 vector 384 chiều(1 dãy gồm 384 số, ko phải 1 số)
    return model.encode(word) 

def sentence_representation_no_position(sentence_words, model):
    embeddings = [get_word_embedding(w, model) for w in sentence_words]
    #chỗ này là lặp qua từng phần tử một trong sentence_word, VD: embedding từ "chó" -> là 1 vector 384 số
    return sum(embeddings) #trả về cộng tổng của tất cả các vector trong sentence_words (cộng vị trí tương ứng cho nhau) 

model_using = SentenceTransformer("intfloat/multilingual-e5-small")

sentence_1 = ['chó', 'cắn', 'người']
sentence_2 = ['người', 'cắn', 'chó']

rep_1 = sentence_representation_no_position(sentence_1, model_using) 
rep_2 = sentence_representation_no_position(sentence_2, model_using)

print(np.allclose(rep_1, rep_2))