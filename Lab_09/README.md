# Lab 09: Sequence Generation using RNN/LSTM and Transformers

## Objective
The objective of this lab is to understand how generative models can be applied to sequential data such as text, time-series, or language sequences. We design and implement simple generative models capable of learning patterns from sequences and generating new sequences.

## Components
### Component I: Sequence Generation using RNN / LSTM
- Developed a character/word-level recurrent neural network using LSTM.
- Model trained on sequential dataset.
- Sequence generated successfully.

### Component II: Transformer Based Sequence Generation
- Developed a simple Transformer model for sequence generation.
- Model uses a positional encoder and transformer block.
- Sequence mapped and predicted correctly based on seed input.

## Dataset Used
```
machine learning models learn patterns from data.
sequence models process data step by step.
recurrent neural networks are designed for sequential tasks.
rnn models maintain hidden states across time steps.
long short term memory networks solve long dependency problems.
lstm uses gates to control information flow.
gru models simplify the lstm architecture.
sequence prediction is useful in many applications.
language modeling predicts the next word in a sentence.
speech recognition processes audio sequences.
time series forecasting predicts future values.
music generation creates new melodies.
generative models learn probability distributions.
they generate new samples similar to training data.
sequence generation is widely used in artificial intelligence.
deep learning improves sequence modeling performance.
```

## Running the Code
Ensure you have PyTorch, Matplotlib, and other common dependancies installed:
```bash
python sequence_generation.py
```

## Output
Outputs generated text models alongside loss curves which predict the model's accuracy, saved neatly inside the `outputs` directory.
