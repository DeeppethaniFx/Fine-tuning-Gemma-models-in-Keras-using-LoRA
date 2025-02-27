# Fine-tune Gemma Models in Keras using LoRA

## Overview

This repository provides a tutorial on fine-tuning Gemma model (gemma2_2b_en) using Low Rank Adaptation (LoRA) with KerasNLP. Gemma is a family of lightweight, state-of-the-art open models developed using the same technology behind the Gemini models.

LoRA significantly reduces the number of trainable parameters, making fine-tuning faster and more memory-efficient while maintaining high-quality model outputs. In this tutorial, we fine-tune the Gemma 2B model using the Databricks Dolly 15k dataset.

## Prerequisites

Before starting, ensure you have:
- Access to Gemma models on Kaggle.
- A Colab runtime with a **T4 GPU** or higher.
- A Kaggle API key for downloading datasets.

## Setup Instructions

### 1. Configure Kaggle API Key
Generate your API key on Kaggle and store it in `kaggle.json`. Then, set the environment variables.

### 2. Install Dependencies
Install Keras, KerasNLP, and other necessary libraries.

### 3. Select Backend
Choose between JAX, TensorFlow, or PyTorch as the backend for running the model.

## Load and Preprocess Dataset

Download and process the Databricks Dolly 15k dataset. This dataset contains human-generated prompt/response pairs for fine-tuning LLMs.

## Load the Gemma Model

Use KerasNLP to load the Gemma model for causal language modeling.

## Fine-tune using LoRA

Enable LoRA with a chosen rank to reduce the number of trainable parameters while maintaining performance. Compile and fine-tune the model using appropriate optimizers and loss functions.
- LoRA Rank = 4
- Total params: 2,617,270,528 (9.75 GB)
- Trainable params: 2,928,640 (11.17 MB)
- Non-trainable params: 2,614,341,888 (9.74 GB)

## Running Inference

After fine-tuning, generate responses based on user queries to observe improvements in model output.

## Results Comparison

### Without LoRA Fine-Tuning
- The model provides generic responses without specific guidance.
- prompt = "What should I do on a trip to Europe?"
- Response:  " I'm going to visit Europe in June. I will be travelling by train. I will be staying with my aunt and uncle. I have to take a plane to get there. I am not sure what the climate is there so I have to bring a jacket. I am going to go to Paris. I'm looking for a place to stay there. I want to visit the Eiffel Tower. I'll also visit the Louvre, Notre Dame Cathedral and other places. I'm also looking for the Eiffel Tower. I'll also visit the Louvre and Notre Dame Cathedral. I'll also visit the Eiffel Tower and Notre Dame Cathedral. I'm also looking at the Eiffel Tower. I'm looking for a place to stay there. I'm going to visit the Eiffel Tower. I'm looking for a place to stay there. I'm going to visit the Eiffel Tower. I'm looking for a place to stay there. I'm going to visit the Eiffel Tower. I'm looking for a place to stay there. I'll also visit the Louvre and Notre Dame Cathedral. "
- When asked about a trip to Europe, the model gives broad, non-specific travel tips.


### With LoRA Fine-Tuning
- The model generates more detailed and context-aware responses.
- prompt = "What should I do on a trip to Europe?"
- Response: " You should visit the Eiffel Tower in Paris, the Leaning Tower of Pisa in Italy, the Colosseum in Rome and the Sagrada Familia in Barcelona."
- Example: After fine-tuning, the model recommends specific destinations and activities for a trip to Europe.



