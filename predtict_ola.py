from transformers import T5ForConditionalGeneration, T5Tokenizer

# Replace 'path_to_your_finetuned_model_directory' with the actual path where your fine-tuned model is saved
finetuned_model_path = 't5_finetuned'

# Load the fine-tuned model and tokenizer
finetuned_model = T5ForConditionalGeneration.from_pretrained(finetuned_model_path)
tokenizer = T5Tokenizer.from_pretrained(finetuned_model_path)

# Set the model to evaluation mode
finetuned_model.eval()

# Example input text
input_text = "I am at home and I want to book a ride to the office using Ola auto"

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate text with max_length parameter
max_length = 90  # Adjust this value according to your desired maximum length
output_ids = finetuned_model.generate(input_ids, max_length=max_length)

# Decode the generated output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated output
print("Generated Output:", output_text)
