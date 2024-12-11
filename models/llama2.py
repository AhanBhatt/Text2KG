from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Load LLaMA 2 model and tokenizer
model_name = "meta-llama/Llama-2-13b-chat-hf"  # Make sure the model is downloaded
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def analyze_text_with_llama(text):
    """Analyze text using LLaMA 2 to extract entities and relationships."""
    prompt = f"""
    Extract entities and their relationships from the following text:
    
    Text:
    {text}
    
    Provide the output in the format:
    Entities: [Entity1, Entity2, ...]
    Relationships: [(Entity1, Entity2, 'Relationship1'), (Entity3, Entity4, 'Relationship2'), ...]
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=512, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

