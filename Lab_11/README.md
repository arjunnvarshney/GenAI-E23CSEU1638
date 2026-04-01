# Lab 11: Fine-Tuning GPT-2 for Real-World Industry Applications

## Objective
The objective of this lab is to fine-tune a pre-trained generative model (GPT-2) for real-world applications. We use transfer learning via fine-tuning to adapt the model to specific business domains to build:
1. A **Product Review Generator** for e-commerce.
2. A **Recipe Instruction Generator** for a food-tech application.

## Components
### Component I: Fine-Tune GPT-2 as a Product Review Generator (E-Commerce)
- Loaded a pre-trained `gpt2` model.
- Evaluated its baseline product review suggestions using zero-shot inference.
- Prepared a dataset containing e-commerce reviews.
- Tokenized the corpus and executed fine-tuning on top of the pre-trained weights for 15 epochs.
- Analyzed the perplexity drops and successfully verified that the fine-tuned completions better align with customer product review conventions.

### Component II: Fine-Tune GPT-2 as a Recipe Instruction Generator (Food-Tech)
- Loaded a pristine `distilgpt2` instance for culinary tasks.
- Generated baseline culinary knowledge before exposing it to the recipe dataset.
- Prepared a food-tech dataset specifying concise cooking methods (marinating, cooking, and serving details).
- Ran gradient updates on the causal masking objective and fine-tuned it over cooking instructions.
- Generated cooking-specific patterns showing strong structural adherence using culinary prompts.

## Dataset Used
- **Component I:** 20 E-commerce Product reviews emphasizing durability, usage cases, battery life, design aesthetics, delivery, and satisfaction.
- **Component II:** Culinary step-by-step paragraphs on preparing butter chicken, pasta carbonara, vegetable stir fry, and chocolate chip cookies.

## Running the Code
Ensure you have `transformers`, `datasets`, `accelerate` and PyTorch via pip:
```bash
# Optional dependency pre-requisites
pip install transformers datasets accelerate 

# Run fine-tuning procedure
python fine_tune_gpt2.py
```

## Output
Review the raw baseline versus fine-tuned output logs within `outputs/fine_tuning_results.txt`. The model successfully demonstrates adaptation toward vocabulary representing domain tasks—eschewing generalized internet text for explicitly targeted e-commerce or food-tech structured output conventions.
