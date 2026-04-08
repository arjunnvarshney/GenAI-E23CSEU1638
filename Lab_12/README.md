# Lab 12: Generative Model with Attention

## Objective
The objective of this lab is to implement a text generation model using an Attention mechanism to improve contextual understanding over sequence predictions. We build a chatbot architecture (seq2seq) that generates contextual replies from user input where attention learns to focus on the key important words inside an encoder sentence.

## Execution Requirements
- Used PyTorch for the Deep Learning implementation.
- Modeled conversational pairings simulating the Cornell Movie Dialogs format.
- Output text and Attention Weights (heatmaps) are actively saved to `outputs/`.

## Architecture Details
- **Preprocessing:** Build word-to-index tracking logic encapsulating PAD, EOS, and SOS bounds.
- **Encoder:** An `nn.Embedding` mapped sequence runs through an `LSTM` tracking context states.
- **Attention Interface:** Learned alignment via dense soft-max mapped context weights over the encoded hidden representations alongside decoder state.
- **Decoder:** Attention-weighted input maps dynamically sequentially to step-by-step generations via `NLLLoss` and `Adam`.

## Expected Scenario Addressed
- *Input:* "how are you" 
- *Output:* "i am fine"
- Model plots correctly mapped attention distributions!

## Project Navigation
```bash
# Run Model Training & Generation
python attention_chatbot.py
```
Check `outputs/results.txt` for chatbot outputs and `outputs/*.png` for the detailed attention heatmaps generated per phrase.
