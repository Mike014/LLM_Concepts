from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

prompt = input("Write your Prompt:")
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=15,
    num_return_sequences=1,
    temperature=0.2,
    top_k=5,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2,
    length_penalty=1.0,
    early_stopping=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# The generator function (typically a method like generate in a Hugging Face model) can accept many other parameters to control the text generation process. Here are some of the most common parameters:

# min_length: The minimum length of the generated text.
# temperature: A temperature value for sampling. Higher values make the text more random, while lower values make the text more deterministic.
# top_k: The number of highest probability tokens to consider during sampling.
# top_p: The cumulative probability threshold for nucleus sampling.
# do_sample: If True, uses sampling; if False, uses greedy search.
# repetition_penalty: A penalty factor to avoid repetition of tokens.
# length_penalty: A penalty factor for the length of the generated sequences.
# early_stopping: If True, stops the generation when an early stopping condition is met.