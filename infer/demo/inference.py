from transformers import AutoModelForCausalLM, AutoTokenizer

def model_fn(model_dir):
  loaded_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  return loaded_model, tokenizer 

def predict_fn(data, model_and_tokenizer):
  model, tokenizer = model_and_tokenizer
  prompt = data.pop("inputs", data)
  inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
  outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
  predictions = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
  return [{"generated_text": predictions}]