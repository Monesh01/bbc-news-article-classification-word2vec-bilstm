import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

## Stopwords from nltk
nltk.download('stopwords')
english_stops = set(stopwords.words('english'))
nltk.download('wordnet')

df_raw = pd.read_csv("BBC_news_dataset.csv")
df = df_raw[['text', 'labels']].copy()
df['text'] = df['text'].str.lower()
df['tokens'] = df['text'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))
df['tokens'].iloc[0]

df['tokens'] = df['tokens'].apply(lambda x: [word for word in x.split() if word not in english_stops])

lemmatizer = WordNetLemmatizer()
df['final_tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])


df = df.sample(frac = 1, random_state = 42).reset_index(drop=True)

w2v_model = Word2Vec(
    sentences = df['final_tokens'],
    vector_size=100,    
    window=5,           
    min_count=1,        
    workers=4,
    sg=1,               
    epochs=50
)
embedding_dim = w2v_model.vector_size
vocab = w2v_model.wv.index_to_key
vocab_size = len(vocab)
word_to_idx = {word: i+2 for i, word in enumerate(vocab)}  

embedding_matrix = np.zeros((vocab_size + 2, embedding_dim))

for word, idx in word_to_idx.items():
    embedding_matrix[idx] = w2v_model.wv[word]

def encode_sentence(tokens, word_to_idx):
    return [word_to_idx[word] if word in word_to_idx else 1 for word in tokens]

encoded_sentences = [
    encode_sentence(sentence, word_to_idx)
    for sentence in df['final_tokens']
] 

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = max(len(s) for s in encoded_sentences)

padded_sentence = pad_sequences(
    encoded_sentences,
    maxlen = max_len,
    padding = "post"
)


encoder = LabelEncoder()
df['encoded_labels'] = encoder.fit_transform(df['labels'])
classes = encoder.classes_


data_config = {
    'word_index' : word_to_idx,
    'classes' : classes.tolist(),
    'max_len' : max_len
}

import json
with open('metadata.json', 'w') as f:
    json.dump(data_config, f)


w2v_model.save('BBC_embeddings')