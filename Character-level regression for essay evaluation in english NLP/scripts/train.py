import torch
import config 
import pandas as pd

from model import NeuralNet
from dataset import dataset
from torch.utils.data import DataLoader


def start_training():
    
    train_df = pd.read_csv(config.path + "train.csv")
    val_df = train_df.iloc[3500:, :]
    train_df = train_df.iloc[:3500, :]
    
    train_texts = train_df["full_text"]
    train_targets = train_df.iloc[:, 2:]
    
    val_texts = val_df["full_text"]
    val_targets = val_df.iloc[:, 2:]
    
    train_dataset = dataset(train_texts, train_targets.values)
    val_dataset = dataset(val_texts.values, val_targets.values)
    
    
    train_dataLoader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataLoader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
    
    model = NeuralNet(in_channels=config.max_char_length, num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = torch.nn.MSELoss()
    
    config.start_training(model, config.num_iterations, optimizer, criterion, train_dataLoader, val_dataLoader, scheduler)
    

if __name__ == "__main__":
    start_training()

