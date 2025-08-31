
# LangGraph Agent Setup
# Run: pip install -U langgraph "langchain[anthropic]"

import getpass
import os
# Use a pipeline as a high-level helper

from transformers import pipeline


pipe = pipeline("text-generation", model="google/gemma-3-270m")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

def generate_introduction(state):
    # Your prompt engineering here
    prompt = f"Write a brief creative introduction for a job application with eye catchy and fun word play. Job: {state['teacher']} Candidate: {state['rinky']}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"draft": generated_text}

if __name__ == "__main__":
    # Example usage:
    state = {
        "teacher": "Software Engineer",
        "rinky": "Rinky"
    }
    result = generate_introduction(state)
    print(result['draft'])

