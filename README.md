# LLM
**Large Language Models (LLM)** are a type of **Deep Learning model** specialized in understanding and generating **natural language**.

They are **based on advanced neural networks**, in particular on **Transformers**, which represent the architecture underlying LLMs.

## Training: Pre-training e Fine-tuning
- **Pre-training**: The model is trained on huge amounts of data to learn general language structures.
- **Fine-tuning**: It is then adapted to specific tasks (e.g. customer service, medicine, finance) with smaller datasets.

## Concept of Prompt and Transformer (Encoder-Decoder)

- The prompt is the input given to the model to get a desired output.
- It is based on the Encoder-Decoder architecture:
  1. **Encoder**: interprets and compresses the input into an internal representation.
  2. **Decoder**: generates the output based on this representation.
  3. **Prompt Engineering**: Optimization of the prompt to get better answers (clear, concise, task-oriented).

## Types of LLM

- **Generic (Auto-regressive)** → Predicts the next token based on training data. Ex: IntelliSense.
- **Instruction-Tuned** → Models optimized to execute specific commands (e.g. "Summarize the text").
- **Dialog-Tuned** → Optimized for interactive conversations, such as ChatGPT or Gemini.


### First create and activate a Conda environment 
  ```bash
   conda create -n llm_env python=3.12
   conda activate llm_env
   ```

### Download libraries
  ```bash
   pip install transformers torch
   ```

### Run the script
  ```bash
   python generate_text.py
   ```
