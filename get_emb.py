import pickle

import numpy as np
import pandas as pd
import torch
from BERT import *
from Bio import SeqIO

from utils import process_sequences

# Check if GPU is available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load the saved model
parameter = {
    'embed_dim': 128,
    'attn_dropout': 0.1,
    'intermediate_size': 1024,
    'inter_act_fn': nn.GELU(),
    'encode_dropout': 0.1,
    'num_heads': 8,
    'hidden_dropout': 0.1,
    'output_dropout': 0.1,
    'num_layers': 8,
    'vocab_size': 8004,
}
model = BertModel(parameter).to(device)
# Load the model's state dictionary
state_dict = torch.load('./data/model.pth', map_location=device)['model_state_dict']
# Remove 'module.bert.' prefix
new_state_dict = {k.replace('module.bert.', ''): v for k, v in state_dict.items()}
# Remove MLM task layers
keys_to_remove = [
    "module.classifier.dense.weight",
    "module.classifier.dense.bias",
    "module.classifier.LayerNorm.weight",
    "module.classifier.LayerNorm.bias",
    "module.classifier.decoder.weight",
    "module.classifier.decoder.bias"
]
new_state_dict = {k: v for k, v in new_state_dict.items() if k not in keys_to_remove}
model.load_state_dict(new_state_dict)
model.eval()  # Set model to evaluation mode

# Load Word2Vec embeddings
with open("./data/embeddings.pkl", 'rb') as file:
    embeddings = pickle.load(file)

# Read sequences from FASTA files
positive_seqs = [str(seq.seq) for seq in SeqIO.parse("./Rice/cd_drought_positive.fasta", "fasta")]
unlabeled_seqs = [str(seq.seq) for seq in SeqIO.parse("./Rice/cd_drought_unlabeled.fasta", "fasta")]

# Process sequences to get token IDs
all_seqs = positive_seqs + unlabeled_seqs
processed_sequences = process_sequences(all_seqs, window_size=3)

# Create token IDs based on the vocabulary
with open("./data/vocab.pkl", 'rb') as file:
    vocab = pickle.load(file)

token_ids = [[vocab.get(token) for token in seq] for seq in processed_sequences]

# Generate embeddings for BERT input
input_vec = []
for token_seq in token_ids:
    embedding = [embeddings[idx] for idx in token_seq]
    embedding = [list(map(float, em)) for em in embedding]
    embedding = torch.tensor(embedding, dtype=torch.float32, device=device)
    input_vec.append(embedding)

# Get labels based on sequences
labels = [1] * len(positive_seqs) + [0] * len(unlabeled_seqs)

# Generate BERT embeddings
bert_emb = []
with torch.no_grad():  # Do not compute gradients
    for emb in input_vec:
        output = model(emb.unsqueeze(1))  # Add batch dimension
        final_emb = torch.mean(output, dim=0).squeeze()  # # Average across the sequence length
        bert_emb.append(final_emb.cpu().numpy())  # Move tensor to CPU and convert to NumPy array

# Average embeddings for every three original sequences
final_bert_emb = []
for i in range(0, len(bert_emb), 3):  # Process every three embeddings
    avg_emb = np.mean(bert_emb[i:i + 3], axis=0)  # Average using NumPy
    final_bert_emb.append(avg_emb)

# Save embeddings and labels to CSV
data = []
for emb, label in zip(final_bert_emb, labels):
    row = [label] + emb.tolist()
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("./Rice/embedding_and_labels.csv", index=False, header=False)


