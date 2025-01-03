import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

# Load dataset
with open('./StoryGenerator/data/children_stories.txt_updated', 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Clean the text
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # Total unique words

with open('./StoryGenerator/lstm/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print(f'max_sequence_len: {max_sequence_len}')

# Randomly sample 10% of the data
np.random.seed(42)  # For reproducibility
sample_size = int(0.10 * len(input_sequences))
sample_indices = np.random.choice(len(input_sequences), sample_size, replace=False)
input_sequences = input_sequences[sample_indices]

# Split into predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
# label = to_categorical(label, num_classes=total_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))  # Embedding layer
model.add(LSTM(256, return_sequences=True))  # First LSTM layer
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(100))  # Second LSTM layer
model.add(Dense(total_words, activation='softmax'))  # Output layer

# model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# model.add(Bidirectional(LSTM(150)))
# model.add(Dense(total_words, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Train the model
history = model.fit(predictors, label, epochs=20, batch_size=64, verbose=1)
model.save('./StoryGeneratorlstm/lstm_bedtime_story_model.h5')

print(history)
print("Model training complete")

