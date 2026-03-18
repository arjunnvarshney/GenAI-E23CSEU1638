import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Define dataset
sentences = [
    "machine learning models learn patterns from data",
    "sequence models process data step by step",
    "recurrent neural networks are designed for sequential tasks",
    "rnn models maintain hidden states across time steps",
    "long short term memory networks solve long dependency problems",
    "lstm uses gates to control information flow",
    "gru models simplify the lstm architecture",
    "sequence prediction is useful in many applications",
    "language modeling predicts the next word in a sentence",
    "speech recognition processes audio sequences",
    "time series forecasting predicts future values",
    "music generation creates new melodies",
    "generative models learn probability distributions",
    "they generate new samples similar to training data",
    "sequence generation is widely used in artificial intelligence",
    "deep learning improves sequence modeling performance"
]

# Task 1 & 2 & 3: Tokenization and input-output pairing
words = set(" ".join(sentences).split())
word2idx = {w: i+1 for i, w in enumerate(sorted(list(words)))}
word2idx['<PAD>'] = 0
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

seq_len = 3  # We use 3 words to predict the next word
X, Y = [], []

for sent in sentences:
    tokens = [word2idx[w] for w in sent.split()]
    for i in range(len(tokens) - seq_len):
        X.append(tokens[i:i+seq_len])
        Y.append(tokens[i+seq_len])

X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# Task 4 Component I: Design LSTM Model
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out[:, -1, :])  # Predict based on the last sequence output
        return out

# Task 4 Component II: Design Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        out = self.fc_out(out[:, -1, :]) # Predict based on last position
        return out

# Task 5 Component I & II: Training

def train_model(model, epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return loss_history

# Initialize Models
embed_dim = 16
hidden_dim = 32

print("\n--- Training LSTM Model ---")
lstm_model = LSTMGenerator(vocab_size, embed_dim, hidden_dim)
lstm_losses = train_model(lstm_model, epochs=150, lr=0.01)

print("\n--- Training Transformer Model ---")
transformer_model = TransformerModel(vocab_size, embed_dim, num_heads=2, hidden_dim=32, num_layers=2)
transformer_losses = train_model(transformer_model, epochs=150, lr=0.01)

# Plot Loss
plt.figure()
plt.plot(lstm_losses, label='LSTM Loss')
plt.plot(transformer_losses, label='Transformer Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Sequence Generators')
plt.legend()
plt.savefig('outputs/training_loss.png')
print("\nLoss plot saved to outputs/training_loss.png")

# Task 6: Generation
def generate_sequence(model, seed_words, num_generate=5):
    model.eval()
    words_seq = seed_words.split()
    generated = list(words_seq)
    
    for _ in range(num_generate):
        # Convert last seq_len words to indices
        seq_tokens = [word2idx.get(w, 0) for w in generated[-seq_len:]]
        # Pad if seed is too short
        while len(seq_tokens) < seq_len:
            seq_tokens.insert(0, 0)
            
        x_input = torch.tensor([seq_tokens], dtype=torch.long)
        with torch.no_grad():
            output = model(x_input)
            predicted_idx = output.argmax(1).item()
            
        predicted_word = idx2word.get(predicted_idx, '<UNK>')
        generated.append(predicted_word)
        
    return " ".join(generated)

seed = "machine learning models"
print(f"\nSeed Text: '{seed}'")

lstm_gen = generate_sequence(lstm_model, seed, num_generate=4)
print(f"LSTM Generated: {lstm_gen}")

trans_gen = generate_sequence(transformer_model, seed, num_generate=4)
print(f"Transformer Generated: {trans_gen}")

# Save generated output to a text file
with open('outputs/generated_sequences.txt', 'w') as f:
    f.write(f"Seed Text: '{seed}'\n\n")
    f.write(f"LSTM Generated sequence:\n{lstm_gen}\n\n")
    f.write(f"Transformer Generated sequence:\n{trans_gen}\n")
    
print("Output saved to outputs/generated_sequences.txt")
