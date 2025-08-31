
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Force CPU usage
device = torch.device("cpu")

# Using a more powerful, instruction-tuned model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    trust_remote_code=True # Required for this model
).to(device)

def generate_introduction(details):
    # A more detailed prompt that uses personal details to create a compelling narrative
    prompt = f"""
    As a creative assistant, write a short, catchy, and professional introductory message (2-3 paragraphs) to a hiring team.

    Use the following details to make it personal and impactful:
    - Candidate's Name: {details['name']}
    - Job Title Applying For: {details['job_title']}
    - Years of Experience: {details['experience']}
    - Key Skills: {details['skills']}
    - Specific Reason for Interest: {details['reason_for_interest']}

    The message should grab the reader's attention, briefly introduce the candidate, highlight their key qualifications, and express genuine enthusiasm for the specific role and company.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        # Increased token limit for a more complete message
        outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return {"draft": generated_text}

if __name__ == "__main__":
    # --- Replace these details with your own! ---
    personal_details = {
        "name": "Rinky",
        "job_title": "Lead Data Scientist",
        "experience": "8 years",
        "skills": "Python, machine learning, deep learning, and cloud platforms like AWS and Azure",
        "reason_for_interest": "I'm incredibly impressed by your company's innovative work in AI-driven healthcare solutions, and I believe my experience in predictive modeling can directly contribute to your upcoming projects."
    }
    
    print("Generating your personalized introduction...\n")
    result = generate_introduction(personal_details)
    print("--- Your Draft Introduction ---\n")
    print(result['draft'])

