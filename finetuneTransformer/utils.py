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
    Calculate perplexity for a list of sequences using a Hugging Face model and tokenizer.
    """
    import torch
    from torch.nn.functional import cross_entropy

    total_log_prob = 0
    total_word_count = 0

    for sequence in sequences:
        # Tokenize using Hugging Face tokenizer
        inputs = tokenizer(sequence, return_tensors="pt", max_length=max_sequence_len, truncation=True)
        input_ids = inputs["input_ids"]

        # Use the model to predict log probabilities
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  # Cross-entropy loss

        # Convert loss to log probability
        total_log_prob += loss.item() * input_ids.size(1)  # Multiply by sequence length
        total_word_count += input_ids.size(1)  # Add number of tokens

    # Compute perplexity
    perplexity = torch.exp(torch.tensor(total_log_prob / total_word_count))
    return perplexity.item()


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

def generate_greedy(model, tokenizer, input_text, max_sequence_len=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_sequence_len,
        do_sample=False  # Ensures greedy decoding
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_beam_search(model, tokenizer, input_text, max_sequence_len=50, num_beams=5):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_sequence_len,
        num_beams=num_beams,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_top_k(model, tokenizer, input_text, max_sequence_len=50, top_k=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_sequence_len,
        do_sample=True,
        top_k=top_k
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_top_p(model, tokenizer, input_text, max_sequence_len=50, top_p=0.9):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_sequence_len,
        do_sample=True,
        top_p=top_p,
        top_k=0  # Disable top-k to focus on top-p
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

