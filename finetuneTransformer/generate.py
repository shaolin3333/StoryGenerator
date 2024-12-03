from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from utils import calculate_perplexity, calculate_bleu_score, generate_beam_search, generate_top_k, generate_top_p, generate_greedy

MAX_SEQUENCE_LEN = 100
MAX_WORDS = 50

# Generate Bedtime Stories
def generate_story(prompt, model, tokenizer, max_length=MAX_SEQUENCE_LEN):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


saved_model_path = "./StoryGenerator/finetuneTransformer/bedtime_story_model"
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForCausalLM.from_pretrained(saved_model_path)

prompt = "Once upon a time in a magical forest"

perplexity_dict = {}

beem_story = generate_greedy(model=model, tokenizer=tokenizer, input_text=prompt, max_sequence_len=MAX_SEQUENCE_LEN)
print("generate by beem search:\n" + beem_story)
sequences = [prompt, beem_story]
# print(sequences)
perplexity = calculate_perplexity(model = model, tokenizer=tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("greedy_search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["greedy_search"] = perplexity
print("greedy_search BLEU Score: " + str(bleu_score))

greed_stroy = generate_beam_search(model=model, tokenizer=tokenizer, input_text=prompt, max_sequence_len=MAX_SEQUENCE_LEN, num_beams=5)
print("generate by greedy search:\n" + greed_stroy)
sequences = [prompt, greed_stroy]
# print(sequences)
perplexity = calculate_perplexity(model = model, tokenizer=tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("beam_search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["beam_search"] = perplexity
print("beam_search BLEU Score: " + str(bleu_score))


top_k_story = generate_top_k(model=model, tokenizer=tokenizer, input_text=prompt,  max_sequence_len=MAX_SEQUENCE_LEN, top_k=50)
print("generate by top k sampling:\n" + top_k_story)
sequences = [prompt, top_k_story]
# print(sequences)
perplexity = calculate_perplexity(model = model, tokenizer=tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("top_k_story search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["top_k"] = perplexity
print("top_k_story search BLEU Score: " + str(bleu_score))


top_p_story = generate_top_p(model=model, tokenizer=tokenizer, input_text=prompt,  max_sequence_len=MAX_SEQUENCE_LEN, top_p=0.9)
print("generate by top p sampling:\n" + top_p_story)
sequences = [prompt, top_p_story]
# print(sequences)
perplexity = calculate_perplexity(model = model, tokenizer=tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("top_p_story search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["top_p"] = perplexity
print("top_p_story search BLEU Score: " + str(bleu_score))
print(perplexity_dict)

# greedy search is a search algorithm that always chooses the most likely next word
def greedy_search(prompt, model, tokenizer, max_length=200):
    inputs_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    iterations = []
    n_steps = 8
    choice_per_step = 5
    with torch.no_grad():
        for _ in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(inputs_ids[0], skip_special_tokens=True)
            output = model(input_ids = inputs_ids)
            
            # select logits of the first batch and the last token and apply softmax
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim = -1, descending=True)
            
            # store tokens with highest probabilities
            for choice_idx in range(choice_per_step):
                token_id = sorted_ids[choice_idx].item()
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%))")
                iteration[f"choice_{choice_idx + 1}"] = token_choice
                
            # append predicted next token to input
            inputs_ids = torch.cat([inputs_ids, sorted_ids[None, 0, None]], dim=-1)
            print("Iteration:", iteration)
            iterations.append(iteration)
    pd.DataFrame(iterations)
    
print("************************Greedy Search:************************")    
#greedy_search(prompt, model, tokenizer)
