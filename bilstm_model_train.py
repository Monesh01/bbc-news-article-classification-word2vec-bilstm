import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers


#w2v_model.save('BBC_embeddings')
w2v_model = Word2Vec.load('BBC_embeddings') 

embedding_layer = tf.keras.layers.Embedding(
    input_dim = vocab_size + 2,
    output_dim = embedding_dim,
    weights=[embedding_matrix],
    trainable=False,  
    mask_zero=True,
    name="word_embedding"
) 

tf.keras.mixed_precision.set_global_policy('mixed_float16')
input_layer = layers.Input(shape = (max_len,), name = 'input', dtype = tf.int32)

lstm_layer = layers.LSTM(
    units = 128,
    return_sequences = False,
    name = 'lstm',
    dropout = 0.3,
    recurrent_dropout = 0,
    dtype = tf.float32
)

output_layer = layers.Dense(5, activation = 'sigmoid', name = 'output', dtype = tf.float32)

bottleneck = layers.Dense(32, name = 'bottleneck', dtype = tf.float32)

x = embedding_layer(input_layer)
x = layers.LayerNormalization(name = "pre_lstm_norm")(x)
x = lstm_layer(x)
x = bottleneck(x)
x = layers.LayerNormalization(name = 'dense_norm')(x)
x = layers.ReLU()(x)
x = output_layer(x)
class_model = tf.keras.Model(inputs = input_layer, outputs = x, name = 'BBC_news_classifier')

class_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

class_model.get_layer('word_embedding').trainable = False   ## Initial training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    patience = 2,
    factor = 0.1,
    min_lr = 0.000001
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True
)

## First 10 epochs of general training 

history = class_model.fit(
    padded_sentence,
    df['encoded_labels'],
    epochs = 10,
    callbacks = [reduce_lr, early_stopping],
    batch_size = 48,
    validation_split = 0.2,
    shuffle = True
) 

# Second Run with gradient flow to the embedding for fine tuning 
class_model.get_layer('word_embedding').trainable = True
class_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = class_model.fit(
    padded_sentence,
    df['encoded_labels'],
    epochs = 10,
    callbacks = [reduce_lr, early_stopping],
    batch_size = 48,
    validation_split = 0.2,
    shuffle = True
)

class_model.save('BBC_best_model') #<- Model with best weights ready for the final; Inference