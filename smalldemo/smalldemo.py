import pandas as pd

train_path = "smalldemo/dataset/train.csv" 

train = pd.read_csv(train_path)
train['text'] = train['text'].fillna('').astype(str)

print("Number of rows and columns in train dataset:", train.shape)
print("\nColumn names:")
print(train.columns)
print("\nData types of columns:")
print(train.dtypes)
print("\nBasic statistics of numerical columns:")
print(train.describe())

# trim dataset
dataset_size = 100
train = train[:dataset_size]
print("\nNumber of rows and columns in train dataset after trimming:", train.shape)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Tokenize train data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])
total_words = len(tokenizer.word_index) + 1
print(f'Total words: {total_words}')

# Prepare input sequences
input_sequences = []
for line in train['text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
print("Example input:")
print(line)
print("<------------------------------------->\n\n")
print("Converted vector:")
print(tokenizer.texts_to_sequences([line])[0])

print(f'\n\nTotal input sequences: {len(input_sequences)}')

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

import numpy as np
# Create predictors and label
predictors, label = input_sequences[:, :-1],input_sequences[:, -1]
label = np.array(label)

print(f'\n\nPredictors shape: {predictors.shape}')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(predictors, label, epochs=100, verbose=1)

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


keywords = ["boy", "duck", "fly"]
seed_text = ' '.join(keywords) 

generated_text = generate_text(seed_text, 100, model, max_sequence_len, tokenizer)
print(generated_text)