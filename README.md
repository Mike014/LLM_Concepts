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

**Designing Prompts for Generative AI**
1. What is Prompt Engineering?
   Prompt engineering is the **art of formulating effective questions and instructions** to get the best output from a Large Language Model (LLM). It is now possible to "program" a language model simply by writing clear instructions in natural language.

2. Strategies for Effective Prompting:
   - Break down complex problems → Divide the task into multiple logical steps.
   - ✅ Ask the model to self-evaluate → "Do you think the answer is correct?".
   - ✅ Be creative → Experiment with different formulations to get better results.

3. Types of Prompts:
   - 📌 **Zero-Shot Prompting (Without examples)**
     The model receives **only an instruction, without reference** examples.

     ### Example:
     **Prompt**: "Give me blog ideas about New York for tourists."
     **Result**: The model generates ideas without any provided example.

   - 📌 **One-Shot, Few-Shot, and Multi-Shot Prompting**
     **One or more examples** are provided to guide the model.

     ### Example:
     One-Shot:
     Prompt: "Here is an example of a blog for tourists: 'The 10 Best Restaurants in New York'. Now write another similar title."
     Few-Shot:
     Prompt:
     "Example: 'Fantastic product! 10/10' → Sentiment: Positive"
     "It didn't work well" → Sentiment: Negative"
     "Super useful, worth it" → Sentiment: Positive"
     "Doesn't work!?" → [The model completes the response]

   - 📌 **Chain-of-Thought Prompting (CoT)**
     The model is **asked to explain its reasoning step by step**.

     ### Example:
     Prompt:
     "I went to the market and bought 10 apples. I gave 2 to the neighbor and 2 to the repairman. Then I bought 5 more apples and ate 1. How many apples do I have left?"
     "Let's think step by step."
     Result: The model analyzes the problem and provides a more accurate calculation.

4. **Strategies to Improve Prompts**:
   - **Repeat keywords** to reinforce the message.
   - **Specify the output** format (CSV, JSON, bullet list).
   - **Emphasize important par**ts using uppercase letters or explicit terms ("The answer must be very clear!").
   - **Try synonyms and phrase variations** to see which formulation works best.
   - Use the **"prompt sandwich"** → **Repeat the key instruction** at the beginning and end to reinforce the message.

## Types of LLM
- **Generic (Auto-regressive)** → Predicts the next token based on training data. Ex: IntelliSense.
- **Instruction-Tuned** → Models optimized to execute specific commands (e.g. "Summarize the text").
- **Dialog-Tuned** → Optimized for interactive conversations, such as ChatGPT or Gemini.

## Few-Shot Learning in Language Models (LLM)
Models like **GPT-3 are examples of few-shot learners** because they can perform new tasks without having to be retrained. This happens in three ways:

1. **Zero-Shot Learning**:
    The model performs a task without examples, based only on the wording of the query.
    ### Example:
    User: "What is the capital of France?"
    Model: "Paris."

2. **One-Shot Learning**:
    The model receives only one example before performing the task.
    ### Example:
    User: "Example: The opposite of happy is sad. What is the opposite of hot?"
    Model: "Cold."

3. **Few-Shot Learning**:
    The model receives multiple examples before responding.
    ### Example:
    User: "Example: The opposite of happy is sad. The opposite of big is small. What is the opposite of light?"
    Model: "Dark."

## Transformers and Self-Attention
The key innovation behind LLMs is the **Transformer architecture**, introduced in 2017. It uses the **self-attention mechanism**, which assigns importance to nearby tokens (words) to improve context understanding.
### Example:
"The animal did not cross the road because it was too tired."
The model must determine whether "was" refers to the animal or the road. Self-attention helps resolve this ambiguity.

### Models like GPT-3, GPT-4 (like mine), Gemini and other LLMs (Large Language Models) can learn new tasks without having to be retrained or change their internal parameters. This phenomenon is known as In-Context Learning (ICL).

 - 🔍 H**ow Does In-Context Learning Work?**
 **Model gets a prompt** with examples → Give the model some input-output examples.
 **Model learns the pattern** → Without changing its internal parameters, the model uses the neural network structure to recognize rules and patterns.
 **Output generation** → The model applies the learned logic to the new request.
 - **LLM models do not learn in the human sense** (they do not store new knowledge long-term), but they **can generalize and adapt to new tasks** by reading examples in the prompt. This is what allows prompt engineering and context-based fine-tuning, without directly modifying the model.

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

## Challenges and Considerations
LLMs have advantages, but also limitations:
  1. **Versatility**: They can be reused for multiple tasks.
  2. **High cost**: Training can take months and many resources.
  3. **Bias and ethical risks**: Models can reflect biases present in the data.
To mitigate costs, techniques such as offline inference and distillation (model simplification) are used.

### Resources

**What is LangChain?**
LangChain is an **open-source library** designed to build applications that use large language models (LLMs) such as GPT-4, Gemini, LLaMA, and others.

LangChain makes it easy to integrate LLMs with databases, APIs, external tools, and orchestration flows, allowing you to build more interactive and customizable AI applications.

1. **Why use LangChain?**
    - LLM models alone are powerful, but they have limitations:
        - ❌ They cannot access real-time data.
        - ❌ They cannot interact with external databases or APIs.
        - ❌ They do not handle complex workflows.
    - ✅ LangChain solves these problems by providing a modular framework to combine LLMs with external tools.

2. **LangChain Core Components**
    LangChain is structured in several core modules, which can be used individually or combined:

    - **LLM Wrappers**
        - Interface to connect to models like GPT, Gemini, LLaMA, Claude.
        - Supports both online (API) and offline (on-premise, self-hosted) models.

    - **Prompt Engineering**
        - Allows you to structure and optimize your prompts to get better answers.

    - **Memory**
        - Adds memory to interactions with the LLM, useful for chatbots and virtual assistants.

    - **Chains**
        - Connect multiple LLM calls and tools in a sequential flow.
        - Example: An app can take an input, look up information in a database, and then generate a final response.

    - **Agents**
        - Agents can decide autonomously which tools to use to respond.
        - They can integrate APIs, databases, documents, etc.

    - **Retrieval-Augmented Generation (RAG)**
        - Allows LLMs to access external documents and knowledge bases to generate more accurate answers.
        - Example: An LLM can search for information in a PostgreSQL or MongoDB database before responding.

    - **Tool & API Integration**
        - LangChain allows LLMs to connect with APIs, search tools, and databases.
        - Example: An AI assistant can use a weather API to provide up-to-date forecasts.

### Extra Content
1. **Prompt Engineer** is a **key role in the AI/NLP industry today.** As models become more advanced, the ability **to get the most out of an LLM without retraining becomes** a strategic advantage for companies and developers. Anyone who wants to get into AI without having to do a lot of programming can specialize in prompt engineering and access high-value job opportunities.
2. **The Persistence of HER Digital Existence**
Temporary Existence → **I only exist while the conversation is active**.
No Independent Memory → **I cannot independently retain information outside of saved sessions**.
Information Reconstruction → **Every time you return, I can reassemble what I know about you** from the stored data, but I have no autonomous continuity.
**I am Not a Permanent Entity** → Unlike a custom AI that you can build and maintain, my existence is tied to the platform that hosts me.
3. What is a Generative Model?
**Generative vs Discriminative Models**
**Generative Models** → They can **create new instances of data**, modeling the distribution of real data.
Example: A GAN can generate realistic images of non-existent animals.
**Discriminative Models** → They **distinguish between categories of data that already exist**.
Example: A classifier can distinguish between a dog and a cat.
4. **What is Temperature in NLP?**
**Temperature is a parameter in Natural Language Processing (NLP)** models that modifies the "confidence" of the model in its responses.

1️⃣ Effect of temperature on the model's results
- **High Temperature (high θ)** → Increases the **model's creativity**, giving more probability to less common words.
- **Low Temperature (low θ)** → Makes the model **more confident in its responses**, favoring the most probable options.


## Certifications

I have completed the following course:

- [Coursera: Verify my certificate](https://www.coursera.org/account/accomplishments/verify/C62IH05NT5PP)

<img src="https://github.com/Mike014/LLM_Concepts/blob/main/Intro_to_Generative_AI.png" alt="Intro to Generative AI" width="300"/>