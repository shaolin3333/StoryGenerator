from transformers import pipeline
from utils import calculate_perplexity, calculate_bleu_score, generate_beam_search, generate_top_k, generate_top_p, generate_greedy

MAX_SEQUENCE_LEN = 100
prompt = "Once upon a time in a magical forest"
generator = pipeline('text-generation', model='gpt2')
# outputs = generator(prompt, max_length=MAX_SEQUENCE_LEN, num_return_sequences=5)

perplexity_dict = {}

beem_story = generate_greedy(model=generator.model, tokenizer=generator.tokenizer, input_text=prompt, max_sequence_len=MAX_SEQUENCE_LEN)
print("generate by beem search:\n" + beem_story)
sequences = [prompt, beem_story]
# print(sequences)
perplexity = calculate_perplexity(model=generator.model, tokenizer=generator.tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("greedy_search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["greedy_search"] = perplexity
print("greedy_search BLEU Score: " + str(bleu_score))

greed_stroy = generate_beam_search(model=generator.model, tokenizer=generator.tokenizer, input_text=prompt, max_sequence_len=MAX_SEQUENCE_LEN, num_beams=5)
print("generate by greedy search:\n" + greed_stroy)
sequences = [prompt, greed_stroy]
# print(sequences)
perplexity = calculate_perplexity(model=generator.model, tokenizer=generator.tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("beam_search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["beam_search"] = perplexity
print("beam_search BLEU Score: " + str(bleu_score))


top_k_story = generate_top_k(model=generator.model, tokenizer=generator.tokenizer, input_text=prompt,  max_sequence_len=MAX_SEQUENCE_LEN, top_k=50)
print("generate by top k sampling:\n" + top_k_story)
sequences = [prompt, top_k_story]
# print(sequences)
perplexity = calculate_perplexity(model=generator.model, tokenizer=generator.tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("top_k_story search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["top_k"] = perplexity
print("top_k_story search BLEU Score: " + str(bleu_score))


top_p_story = generate_top_p(model=generator.model, tokenizer=generator.tokenizer, input_text=prompt,  max_sequence_len=MAX_SEQUENCE_LEN, top_p=0.9)
print("generate by top p sampling:\n" + top_p_story)
sequences = [prompt, top_p_story]
# print(sequences)
perplexity = calculate_perplexity(model=generator.model, tokenizer=generator.tokenizer, sequences=sequences, max_sequence_len=MAX_SEQUENCE_LEN)
print("top_p_story search Perplexity: " + str(perplexity))
bleu_score = calculate_bleu_score(sequences)
perplexity_dict["top_p"] = perplexity
print("top_p_story search BLEU Score: " + str(bleu_score))
print(perplexity_dict)
