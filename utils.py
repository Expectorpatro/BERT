"""
=====================================
@author: Expector
@time: 2024/4/28:下午7:35
@email: 10322128@stu.njau.edu.cn
@IDE: PyCharm
=====================================
"""
# Function to process sequences into window_size-grams
def process_sequences(sequences, window_size):
    processed_sequences = []
    basic_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    for seq in sequences:
        # Skip sequences containing non-standard amino acids
        if any(amino_acid not in basic_amino_acids for amino_acid in seq):
            continue

        length = len(seq)
        # Generate n-grams
        for start in range(window_size):
            processed_seq = ['cls'] + [seq[i:i + window_size] for i in range(start, length - window_size + 1, window_size)] + ['sep']
            processed_sequences.append(processed_seq)

    return processed_sequences

def write_lists_to_file(data, file_path):
    with open(file_path, 'w') as file:
        for sublist in data:
            line = ' '.join(map(str, sublist)) + '\n'
            file.write(line)


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip('\t').split() for line in lines]
    return lines



