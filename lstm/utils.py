import numpy as np
import re
import string
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def beam_search(seed_text, model, tokenizer, max_sequence_len, beam_width=3, max_words=20):
    sequences = [(seed_text, 0)]  # List of tuples (sequence, score)
    for _ in range(max_words):
        all_candidates = []
        for seq, score in sequences:
            token_list = tokenizer.texts_to_sequences([seq])[0]
            token_list = np.pad(token_list, (max_sequence_len - len(token_list) - 1, 0), mode='constant')
            token_list = token_list[-(max_sequence_len - 1):]
            token_list = np.expand_dims(token_list, axis=0)
            
            predicted_probs = model.predict(token_list, verbose=0)[0]
            # Consider top beam_width predictions
            top_indices = np.argsort(predicted_probs)[-beam_width:]
            for index in top_indices:
                candidate_word = tokenizer.index_word.get(index, "")
                new_seq = seq + " " + candidate_word
                new_score = score - np.log(predicted_probs[index])  # Log for numerical stability
                all_candidates.append((new_seq, new_score))
        
        # Sort candidates by score (ascending) and retain top beam_width
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
    
    # Return the best sequence
    return sequences[0][0]

def greedy_search(seed_text, model, tokenizer, max_sequence_len, max_words=20):
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = np.pad(token_list, (max_sequence_len - len(token_list) - 1, 0), mode='constant')
        token_list = token_list[-(max_sequence_len - 1):]
        token_list = np.expand_dims(token_list, axis=0)
        
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + predicted_word
    
    return seed_text

def top_k_sampling(seed_text, model, tokenizer, max_sequence_len, k=10, max_words=20):
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = np.pad(token_list, (max_sequence_len - len(token_list) - 1, 0), mode='constant')
        token_list = token_list[-(max_sequence_len - 1):]
        token_list = np.expand_dims(token_list, axis=0)
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        top_indices = np.argsort(predicted_probs)[-k:]
        top_probs = predicted_probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)  # Normalize probabilities
        
        # Sample from the top-k indices
        chosen_index = np.random.choice(top_indices, p=top_probs)
        predicted_word = tokenizer.index_word.get(chosen_index, "")
        seed_text += " " + predicted_word
    
    return seed_text

def top_p_sampling(seed_text, model, tokenizer, max_sequence_len, p=0.9, max_words=20):
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = np.pad(token_list, (max_sequence_len - len(token_list) - 1, 0), mode='constant')
        token_list = token_list[-(max_sequence_len - 1):]
        token_list = np.expand_dims(token_list, axis=0)
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        sorted_indices = np.argsort(predicted_probs)[::-1]
        sorted_probs = predicted_probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Select top-p indices
        top_p_indices = sorted_indices[cumulative_probs <= p]
        if len(top_p_indices) == 0:  # In case p is very low
            top_p_indices = sorted_indices[:1]
        top_p_probs = predicted_probs[top_p_indices]
        top_p_probs = top_p_probs / np.sum(top_p_probs)  # Normalize probabilities
        
        # Sample from the top-p indices
        chosen_index = np.random.choice(top_p_indices, p=top_p_probs)
        predicted_word = tokenizer.index_word.get(chosen_index, "")
        seed_text += " " + predicted_word
    
    return seed_text

def calculate_perplexity(model, tokenizer, sequences, max_sequence_len):
    """
    Calculate perplexity for a list of sequences.
    """
    total_log_prob = 0
    total_word_count = 0

    for sequence in sequences:
        tokenized_sequence = tokenizer.texts_to_sequences([sequence])[0]
        for i in range(1, len(tokenized_sequence)):
            input_seq = tokenized_sequence[:i]
            input_seq = np.pad(input_seq, (max_sequence_len - len(input_seq), 0), mode='constant')
            input_seq = np.expand_dims(input_seq, axis=0)

            predicted_probs = model.predict(input_seq, verbose=0)[0]
            target_word = tokenized_sequence[i]
            word_prob = predicted_probs[target_word]

            total_log_prob += np.log(word_prob)
            total_word_count += 1

    perplexity = np.exp(-total_log_prob / total_word_count)
    return perplexity

def calculate_bleu_score(sequences):
    """
    Calculate BLEU score for a list of generated texts.
    """
    bleu_scores = []
    for reference, generated in zip(sequences[0], sequences[1]):
        reference = [reference.split()]  # Reference texts should be a list of tokenized sentences
        generated = generated.split()  # Tokenize generated text
        score = sentence_bleu(reference, generated, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(score)
    
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    return average_bleu

