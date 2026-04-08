import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# 1. Dataset & Preprocessing
# A miniature version of conversational pairs simulating Cornell Movie Dialogs dataset
pairs = [
    ("hello", "hi, how are you?"),
    ("hi", "hello there!"),
    ("how are you", "i am fine"),
    ("what is your name", "i am a chatbot"),
    ("good morning", "good morning to you"),
    ("bye", "goodbye"),
    ("thank you", "you are welcome")
]

# Vocabulary building
PAD_token = 0
SOS_token = 1
EOS_token = 2

class Vocab:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.replace('?', '').replace(',', '').replace('!', '').split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

vocab = Vocab()
for pair in pairs:
    vocab.add_sentence(pair[0])
    vocab.add_sentence(pair[1])

def tensorFromSentence(vocab, sentence):
    words = sentence.replace('?', '').replace(',', '').replace('!', '').split(' ')
    indexes = [vocab.word2index[w] for w in words]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# 2. Model: Encoder, Attention, Decoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        
        # Calculate attention weights
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

# 3. Training setup
hidden_size = 64
encoder = EncoderRNN(vocab.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, vocab.n_words)

learning_rate = 0.01
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

epochs = 150
max_length = 10

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

print("Starting training...")
for epoch in range(1, epochs + 1):
    total_loss = 0
    for pair in pairs:
        input_tensor = tensorFromSentence(vocab, pair[0])
        target_tensor = tensorFromSentence(vocab, pair[1])
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
        total_loss += loss
        
    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(pairs):.4f}")

# 4. Evaluation & Attention Visualization
def evaluate(encoder, decoder, sentence, max_length=max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    out_sentence = ' '.join([w for w in output_words if w != '<EOS>'])
    print(f"\nInput: '{input_sentence}'")
    print(f"Output: '{out_sentence}'")
    
    # Save attention plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    # x is input, y is output
    input_words = input_sentence.replace('?', '').replace(',', '').replace('!', '').split(' ') + ['<EOS>']
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f"Attention Weights for '{input_sentence}'")
    plot_path = f"outputs/attention_{input_sentence.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Attention plot successfully saved to {plot_path}")
    return out_sentence

# Test case mentioned in the lab spec
with open('outputs/results.txt', 'w') as f:
    f.write("=== Generative Model with Attention ===\n")
    out_1 = evaluateAndShowAttention("how are you")
    f.write(f"Input: 'how are you'\nOutput: '{out_1}'\n\n")
    
    out_2 = evaluateAndShowAttention("hello")
    f.write(f"Input: 'hello'\nOutput: '{out_2}'\n\n")
