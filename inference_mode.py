import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import re
import nltk
import json
import time

st = time.perf_counter()
#@tf.function
def model_call(input_sequence):
    return model(input_sequence)

def tokenization(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]', '',str(sentence))
    return sentence.split()

def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence = [lemmatizer.lemmatize(x) for x in sentence if x not in english_stops]
    return [word_to_idx[word] if word in word_to_idx else 1 for word in sentence]

model = tf.keras.models.load_model("BBC_best_model")

with open('metadata.json', 'r') as f:
    infer = json.load(f)
word_to_idx, classes, max_len = infer['word_index'], infer['classes'], infer['max_len']

nltk.download('stopwords')
english_stops = set(stopwords.words('english'))

sentence = "I love cricket, so i watch all cricket match of team India"  # input("Enter a News")      
sentence = tokenization(sentence)
tokens = lemmatization(sentence)
pad_tokens = pad_sequences(
    [tokens],
    maxlen = max_len,
    padding = "post"
)
output = model_call(pad_tokens)
Final_output = classes[tf.argmax(output[0]).numpy()]
prob = tf.reduce_max(output[0])
prob = tf.cast(prob, tf.float32)
print(f"The Predicted Class of the given News is : {Final_output} with Probability of {prob * 100:.2f} %")
elapsed= time.perf_counter() - st
print(f"Time taken the whole process is : {elapsed:.6f} seconds")