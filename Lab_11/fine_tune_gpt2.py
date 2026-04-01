import torch, math, os, sys
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling, set_seed)
from datasets import Dataset

# Setup to save outputs
os.makedirs("outputs", exist_ok=True)
output_file = "outputs/fine_tuning_results.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(output_file)

# -------------------------------------------------------------
# Component I: Fine-Tune GPT-2 as a Product Review Generator
# -------------------------------------------------------------
print("Starting Lab 11 - Fine-Tuning GPT-2\n")
print("-" * 50)
print("Component I: E-Commerce Product Reviews")
print("-" * 50)

set_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_length=max_length, temperature=0.8,
            top_k=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).replace('\n', ' ')

review_prompts = [
    'This product is',
    'I bought this phone and',
    'The quality of this item',
]

print('\n=== BASELINE REVIEWS (Before Fine-Tuning) ===')
baseline = {}
for p in review_prompts:
    baseline[p] = generate_text(model, tokenizer, p)
    print(f'Prompt: {p}\nOutput: {baseline[p]}\n')

corpus = [
    'this phone has an amazing battery life and the camera quality is outstanding for the price.',
    'i bought this laptop for college and it handles all my assignments and coding projects perfectly.',
    'the sound quality of these headphones is incredible with deep bass and clear vocals.',
    'this smartwatch tracks my steps accurately and the heart rate monitor is very reliable.',
    'great wireless earbuds with noise cancellation that blocks out all background sound.',
    'the keyboard feels very comfortable for long typing sessions and the backlight is a nice touch.',
    'this portable charger saved me during travel and it charges my phone three times on a single charge.',
    'the tablet screen is bright and colorful which makes watching movies a great experience.',
    'i love this fitness tracker because it motivates me to reach my daily exercise goals.',
    'this bluetooth speaker is compact but delivers surprisingly loud and clear audio.',
    'the delivery was fast and the product was packed securely with no damage at all.',
    'excellent value for money and the build quality feels premium despite the affordable price.',
    'the customer service team was very helpful when i had questions about the product features.',
    'this camera takes stunning photos in low light and the video recording quality is very smooth.',
    'i have been using this product for three months and it still works perfectly like day one.',
    'the design is sleek and modern and it looks great on my desk next to my other gadgets.',
    'easy to set up right out of the box and the instructions were clear and simple to follow.',
    'highly recommend this product to anyone looking for quality and reliability at a fair price.',
    'the software updates keep adding new features which makes this purchase even more worthwhile.',
    'best purchase i made this year and i would definitely buy from this brand again.',
]

dataset = Dataset.from_dict({'text': corpus})
tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True,
    max_length=128, padding='max_length'), batched=True, remove_columns=['text'])
split = tokenized.train_test_split(test_size=0.15, seed=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./outputs/gpt2-reviews', num_train_epochs=15,
    per_device_train_batch_size=4, learning_rate=5e-5,
    weight_decay=0.01, warmup_steps=50, eval_strategy='epoch',
    logging_steps=10, save_strategy='no',
    fp16=torch.cuda.is_available(),
    report_to="none" # Disable logging to external services
)

print('\nStarting fine-tuning for Component I...')
trainer = Trainer(model=model, args=training_args,
    train_dataset=split['train'], eval_dataset=split['test'],
    data_collator=data_collator)

trainer.train()

eval_res = trainer.evaluate()
print(f'\nPerplexity after training (Reviews): {math.exp(eval_res["eval_loss"]):.2f}')

print('\n=== FINE-TUNED REVIEWS (After Fine-Tuning) ===')
for p in review_prompts:
    ft_out = generate_text(model, tokenizer, p)
    print(f'Prompt: {p}')
    print(f'  Baseline:   {baseline[p][:120]}')
    print(f'  Fine-Tuned: {ft_out[:120]}\n')


# -------------------------------------------------------------
# Component II: Fine-Tune GPT-2 as a Recipe Instruction Generator
# -------------------------------------------------------------
print("\n" + "-" * 50)
print("Component II: Food-Tech Recipe Instructions")
print("-" * 50)

# Load a fresh GPT-2
tokenizer2 = GPT2Tokenizer.from_pretrained('distilgpt2')
model2 = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer2.pad_token = tokenizer2.eos_token
model2.config.pad_token_id = tokenizer2.eos_token_id

recipe_prompts = [
    'To make butter chicken',
    'For pasta carbonara',
    'To prepare a chocolate cake',
]

print('\n=== BASELINE RECIPES (Before Fine-Tuning) ===')
baseline2 = {}
for p in recipe_prompts:
    baseline2[p] = generate_text(model2, tokenizer2, p)
    print(f'Prompt: {p}\nOutput: {baseline2[p]}\n')

recipes = [
    'to make butter chicken start by marinating chicken pieces in yogurt with turmeric chili powder and garam masala for one hour.',
    'heat butter in a pan and fry onions until golden brown then add ginger garlic paste and cook for two minutes.',
    'add tomato puree and cook on low heat for ten minutes until the oil separates from the masala.',
    'add the marinated chicken and cook on medium heat for fifteen minutes until fully cooked.',
    'finish with fresh cream and kasuri methi and serve hot with naan or steamed rice.',
    'for pasta carbonara boil spaghetti in salted water until al dente and reserve half cup of pasta water.',
    'fry diced pancetta in olive oil until crispy and set aside.',
    'whisk together egg yolks parmesan cheese and black pepper in a bowl.',
    'toss the hot pasta with pancetta and remove from heat then quickly stir in the egg mixture.',
    'the residual heat will cook the eggs into a creamy sauce and serve immediately with extra parmesan.',
    'to prepare vegetable stir fry heat sesame oil in a wok on high heat.',
    'add sliced bell peppers broccoli florets and snap peas and toss for three minutes.',
    'pour in soy sauce oyster sauce and a pinch of sugar and stir well.',
    'add minced garlic and ginger and cook for one more minute until fragrant.',
    'serve the stir fry over steamed jasmine rice and garnish with sesame seeds.',
    'for chocolate chip cookies cream together butter and sugar until light and fluffy.',
    'beat in eggs one at a time then add vanilla extract and mix well.',
    'fold in flour baking soda and salt then gently stir in chocolate chips.',
    'scoop dough onto a baking sheet and bake at 180 degrees for twelve minutes until golden.',
    'let cookies cool on the tray for five minutes before transferring to a wire rack.',
]

dataset2 = Dataset.from_dict({'text': recipes})
tok2 = dataset2.map(lambda x: tokenizer2(x['text'], truncation=True,
    max_length=128, padding='max_length'), batched=True, remove_columns=['text'])
split2 = tok2.train_test_split(test_size=0.15, seed=42)

collator2 = DataCollatorForLanguageModeling(tokenizer=tokenizer2, mlm=False)

args2 = TrainingArguments(
    output_dir='./outputs/gpt2-recipes', num_train_epochs=15,
    per_device_train_batch_size=4, learning_rate=5e-5,
    weight_decay=0.01, warmup_steps=50, eval_strategy='epoch',
    logging_steps=10, save_strategy='no',
    fp16=torch.cuda.is_available(),
    report_to="none" # Disable logging to external services
)

print('\nStarting fine-tuning for Component II...')
trainer2 = Trainer(model=model2, args=args2,
    train_dataset=split2['train'], eval_dataset=split2['test'],
    data_collator=collator2)

trainer2.train()

eval2 = trainer2.evaluate()
print(f'\nPerplexity after training (Recipes): {math.exp(eval2["eval_loss"]):.2f}')

print('\n=== FINE-TUNED RECIPES (After Fine-Tuning) ===')
for p in recipe_prompts:
    ft = generate_text(model2, tokenizer2, p)
    print(f'Prompt: {p}')
    print(f'  Baseline:   {baseline2[p][:120]}')
    print(f'  Fine-Tuned: {ft[:120]}\n')

print("\nLab 11 execution completed! Results stored in 'outputs/fine_tuning_results.txt'")
