from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
print(generator("Once upon a time", max_length=100, num_return_sequences=5))