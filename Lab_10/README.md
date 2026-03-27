# Lab 10: Sequential Data Generation using RNN/LSTM and Transformers

## Objective
The objective of this lab is to implement a generative model capable of learning patterns from sequential data and generating new sequences. We explore sequence generation using deep learning architectures such as Recurrent Neural Networks (LSTM) and Transformer models.

## Components
### Component I: RNN / LSTM Based Sequential Data Generation
- Loaded and preprocessed the sequential dataset.
- Performed word-level tokenization.
- Designed an LSTM generative model.
- Trained the model and successfully generated sequences based on seed words.

### Component II: Transformer Based Sequential Data Generation
- Utilized a Transformer model for text sequence modeling.
- Created positional encodings to allow the attention mechanism to be sequence-aware.
- Trained the model to map sequence-to-sequence generation perfectly for short sequences.

## Dataset Used
```
artificial intelligence systems learn patterns from data.
sequence models process information step by step.
recurrent neural networks are useful for sequence prediction.
lstm networks handle long term dependencies.
deep learning models improve sequence learning.
generative models create new samples from learned patterns.
language models predict the next word in a sentence.
sequence generation is used in chatbots and assistants.
machine learning helps computers learn automatically.
training data improves model accuracy.
neural networks simulate human brain structures.
optimization algorithms improve learning efficiency.
technology is transforming modern education.
online learning platforms use artificial intelligence.
students benefit from intelligent tutoring systems.
automation improves productivity and decision making.
```

## Running the Code
Ensure you have PyTorch, Matplotlib, and other common dependencies installed:
```bash
python sequence_generation.py
```

## Output
Outputs the generated text strings directly to `outputs/generated_sequences.txt` alongside empirical training loss plots saved within `outputs/training_loss.png`.
