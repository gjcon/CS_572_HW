# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch.nn.functional as F

# --- DO NOT CHANGE THIS SEED ---
# Set random seed for reproducibility and scoring
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_CLASS = 3   #a, b, c parameters
DROP_RATE = 0.2


# Training data
def sine(a,b,c,x):
    x = torch.tensor(x, dtype=torch.float32)
    sine = a * np.sin(b * x + c)
    return torch.tensor(sine, dtype=torch.float32)

# Loss function
def loss(ytrue, ypred):
      return torch.mean((ypred - ytrue)**2)

class my_nn(nn.Module):
    def __init__(self, data_size, embed_dim, hidden_dim, num_class, dropout=0.5):
        super(my_nn, self).__init__()

        self.embedding = nn.EmbeddingBag(data_size, embed_dim, sparse=False, mode="mean")
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, data, offsets):
        # 1. Apply embedding layer
        embedded = self.embedding(data, offsets)

        # 2. Pass through hidden layer and sin(x) activation
        hidden_out = torch.sin(self.hidden(embedded))

        # 3. Apply dropout
        hidden_out = self.dropout(hidden_out)

        # 4. Pass through the output layer
        output = self.output(hidden_out)
        # --- END YOUR CODE ---

        return output
    
# Inititate model
model = my_nn(data_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_class=NUM_CLASS, dropout=DROP_RATE)

# Training Hyperparameters (Provided)
learning_rate = 1e-4
num_epochs = 100

# --- YOUR CODE HERE (~2 Lines)---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


