import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

class WikitextDataset(Dataset):
    def __init__(self, tokenizer, file_path, dataset_type,seq_len=512):
        # Open the file and read the contents
        if os.path.isfile("data/dataset_" + dataset_type + "_cache.dat"):
            all_tokens = torch.load("data/dataset_" + dataset_type + "_cache.dat") # contains all tokens
        else:
            # regenerate the data for entire dataset
            # read all word tokens, then break them into seq_len
            with open(file_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            all_line_tokens = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                    line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(all_lines))
            
            all_tokens = torch.tensor([index for line in all_line_tokens for index in line], dtype=torch.long)
            torch.save(all_tokens, "data/dataset_" + dataset_type + "_cache.dat")
        num_sequences = (all_tokens.size(0) // (seq_len+1)) * (seq_len+1)
        self.data = all_tokens.narrow(0, 0, num_sequences).view(-1, (seq_len+1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens.cuda()
