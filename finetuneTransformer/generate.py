from transformers import AutoTokenizer, AutoModelForCausalLM



# Generate Bedtime Stories
def generate_story(prompt, model, tokenizer, max_length=200):
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


saved_model_path = "./finetuneTransformer/output/bedtime_story_model"
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForCausalLM.from_pretrained(saved_model_path)

prompt = "A boy had a duck friend who dreamed of becoming human. One day, the duck said"
print(generate_story(prompt, model, tokenizer))