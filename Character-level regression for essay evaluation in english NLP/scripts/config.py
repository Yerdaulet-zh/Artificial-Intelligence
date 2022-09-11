import torch

from tqdm import tqdm

path = "../data/feedback-prize-english-language-learning/" # You should download the dataset and save it in the data folder!
encoder_path = "../encoder/binarizer_encoder.joblib"

device = ("cuda" if torch.cuda.is_available() else "cpu")

train_batch_size = 400
val_batch_size = 30
max_char_length = 6044
learning_rate = 0.001
num_classes = 6
num_iterations = 150


characters = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", ",", ":", ";", "?", "!", "$", "&", "(", ")", "[", "]", "{", "}", '""', "''", "`", "'", "#", "@", "%", "/", "|", "'\'", "-", "+", "_", "=", "'", "^", "~", "№", " ", 
] # len == 93


def train(model, criterion, optimizer, scheduler, dataLoader):
    model.train()
    batch_loss, count_comp = 0, 0 
    for texts, targets in tqdm(dataLoader, position=0, total=len(dataLoader)):
        optimizer.zero_grad()
        texts = texts.to(device)
        targets = targets.to(device)

        predicted = model(texts)
        loss = criterion(predicted, targets)
        
        count_comp += 1
        batch_loss += loss
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    
    final_loss = batch_loss / count_comp
    return final_loss


def val(model, criterion, dataLoader):
    with torch.no_grad():
        model.eval()
        batch_loss, count_comp = 0, 0 
        for texts, targets in tqdm(dataLoader, position=0, total=len(dataLoader)):
            texts = texts.to(device)
            targets = targets.to(device)

            predicted = model(texts)

            loss = criterion(predicted, targets)

            count_comp += 1
            batch_loss += loss
        
        final_loss = batch_loss / count_comp
        return final_loss


def save_best_model(epoch, model, train_loss, val_loss, min_val_loss):
    if min_val_loss > val_loss.item():
        torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
            }, "../models/best_model.pt")
        min_val_loss = val_loss
        print("\nThe best model saved!")
        
        
def start_training(model, num_iter, optimizer, criterion, train_dataLoader, val_dataLoader, scheduler=None):
    min_val_loss = 100
    for epoch in range(num_iter):
        train_loss = train(model, criterion, optimizer, scheduler, train_dataLoader)
        print(f"Epoch: {epoch}/{num_iter} | Train MSE loss: {train_loss}")
        val_loss = val(model, criterion, train_dataLoader)
        print(f"Epoch: {epoch}/{num_iter} | Validation MSE loss: {val_loss}")
        
        save_best_model(epoch, model, train_loss, val_loss, min_val_loss)
    
    torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
        }, "../models/last_model.pt")
    print("\nModel saved successfully")
