import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

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

# Keywords for text generation
keywords = ["boy", "cat"]
seed_text = ' '.join(keywords)  # Seed text with keywords

model = load_model('smalldemo/lstm_model.h5')
with open('smalldemo/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Generate text based on keywords
generated_text = generate_text(seed_text, 100, model, 1024, tokenizer)
print(generated_text)