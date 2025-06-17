"""
=====================================
@author: Expector
@time: 2024/4/28:下午6:09
@email: 10322128@stu.njau.edu.cn
@IDE: PyCharm
=====================================
"""
import pickle

import gensim
import numpy as np
from gensim.models import Word2Vec

from utils import *


class MyCallback(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch: {self.epoch} completed.")


def word2vec_embeddings(species_path="./Rice/3_grams.txt", 
                        save_path="./Rice/static_embeddings.txt", 
                        vocab_file="./Rice/vocab.pkl", 
                        embedding_file="./Rice/embeddings.pkl", 
                        token_id_file="./Rice/token_ids.pkl"):
    """
    Train a Word2Vec model using input text data and save the resulting embeddings, 
    vocabulary, and token IDs.

    Parameters:
    species_path : Path to the input text file containing proteins' n_gram data.
    save_path : Path to save the static embeddings in Word2Vec format.
    vocab_file : Path to save the vocabulary dictionary as a pickle file.
                    Maps n_grams to their corresponding indices.
    embedding_file : Path to save the embedding dictionary as a pickle file.
                    Maps indices to their corresponding embedding vectors.
    token_id_file : Path to save the token IDs as a pickle file.
                        Contains lists of token IDs for each protein in the input file.
    """
    
    # Read each line from the text file into a list
    lines = read_file_to_list(species_path)

    # Train the Word2Vec model
    model = Word2Vec(lines, vector_size=128, window=5, epochs=30, negative=20, workers=30, callbacks=[MyCallback()])
    model.wv.save_word2vec_format(save_path)  # Save the trained model in Word2Vec format

    # Build vocabulary and static embedding dictionaries
    vocab_dict, embedding_dict = {}, {}
    with open(save_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header line
        max_idx = len(lines) - 1
        for idx, line in enumerate(lines):
            # Split the line into word and its corresponding embedding
            word, embedding = line.strip('\t').split()[0], line.strip('\t').split()[1:]
            vocab_dict[word] = idx  # Map word to its index
            embedding_dict[idx] = embedding  # Map index to embedding
            
        # Add special tokens for mask and pad
        vocab_dict['mask'] = max_idx + 1
        vocab_dict['pad'] = max_idx + 2
        np.random.seed(2023)
        embedding_dict[max_idx + 1] = list(np.random.randn(128))  # Random embedding for 'mask'
        np.random.seed(2024)
        embedding_dict[max_idx + 2] = list(np.random.randn(128))  # Random embedding for 'pad'

    # Save the vocabulary and embedding dictionaries as pickle files
    with open(vocab_file, 'wb') as file:
        pickle.dump(vocab_dict, file)
    with open(embedding_file, 'wb') as file:
        pickle.dump(embedding_dict, file)

    # Build the token ID list for the species
    lines = read_file_to_list(species_path)
    token_ids = []
    for line in lines:
        if len(line) <= 323:
            # Convert words to their corresponding token IDs using the vocabulary dictionary
            token_id = [vocab_dict[x] for x in line]
            token_ids.append(token_id)
    
    # Save the token IDs as a pickle file
    with open(token_id_file, 'wb') as file:
        pickle.dump(token_ids, file)


if __name__ == "__main__":
    from Bio import SeqIO

    # 将蛋白质序列转换为3_grams并且做好3_mer的分割
    fasta_file = "./Rice/cd_proteins.fasta"
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    processed_seqs = process_sequences(seqs, 3)
    write_lists_to_file(processed_seqs, "./Rice/3_grams.txt")
    word2vec_embeddings(species_path="./Rice/3_grams.txt", 
                        save_path="./data/static_embeddings.txt", 
                        vocab_file="./data/vocab.pkl", 
                        embedding_file="./data/embeddings.pkl", 
                        token_id_file="./data/token_ids.pkl")

