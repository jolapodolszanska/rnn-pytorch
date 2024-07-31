import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
import pickle

# Wczytywanie osadzeń GloVe
with open('glove.840B.300d.pkl', 'rb') as f:
    glove_embeddings = pickle.load(f)

# Wczytywanie danych
df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])

# Czyszczenie danych
def clean_sentences(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return text

df['review'] = df['review'].apply(clean_sentences)

# Tokenizacja
tokenizer = get_tokenizer("basic_english")
vocab = {}
for text in df['review']:
    for word in tokenizer(text):
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
vocab = {word: i+1 for i, (word, _) in enumerate(vocab)}  # +1 aby zarezerwować 0 na padding

def text_to_sequence(text):
    return [vocab.get(word, 0) for word in tokenizer(text)]

df['review'] = df['review'].apply(text_to_sequence)

# Padding
max_len = max(len(seq) for seq in df['review'])
X_data = np.zeros((len(df), max_len), dtype=int)
for i, seq in enumerate(df['review']):
    X_data[i, :len(seq)] = np.array(seq[:max_len])

y_data = df['sentiment'].values

X_train, X_val, Y_train, Y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

# Przygotowanie macierzy osadzeń
embedding_dim = 300
vocab_size = len(vocab) + 1

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Dataset i DataLoader
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

train_dataset = IMDBDataset(X_train, Y_train)
val_dataset = IMDBDataset(X_val, Y_val)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Definicja modelu RNN (LSTM) z wczytanymi osadzeniami GloVe
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Zamrożenie osadzeń

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Ostatni ukryty stan
        x = self.fc(x)
        return self.softmax(x)

model = SentimentRNN(vocab_size, embedding_dim, 64, 2, 2)

# Trening modelu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')

# Funkcje do rysowania wykresów
def plot_graph(train_data, val_data, label):
    plt.plot(train_data, label=f'training {label}')
    plt.plot(val_data, label=f'validation {label}')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(f'{label} vs epochs')
    plt.show()

# Wykresy dla strat i dokładności
plot_graph(train_losses, val_losses, 'loss')
plot_graph(val_accuracies, val_accuracies, 'accuracy')

# Zapis modelu
torch.save(model.state_dict(), 'sentiment_rnn_model.pth')

# Ewaluacja modelu
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in val_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(val_loader)
test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%')
