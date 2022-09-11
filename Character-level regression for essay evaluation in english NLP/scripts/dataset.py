import torch
import joblib
import numpy as np

from config import encoder_path, max_char_length
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, full_text, targets):
        super(dataset, self).__init__()
        self.full_text = full_text
        self.targets = targets
        self.binarizer_encoder = joblib.load(encoder_path)
    
    def __len__(self):
        return len(self.full_text)
    
    def __getitem__(self, x):
        final_binarized_encoded_text = []
        
        text = self.full_text[x]
        chars = [ch for ch in text]
        bin_text = self.binarizer_encoder.transform(chars)
        length = bin_text.shape[0]

        zeros = np.zeros((max_char_length-length, 93))
        encoded_text = np.concatenate((bin_text, zeros))
        final_binarized_encoded_text.append(encoded_text)
        
        final_binarized_encoded_text= torch.FloatTensor(final_binarized_encoded_text[0])
        targets = torch.FloatTensor(self.targets[x])
        return final_binarized_encoded_text, targets