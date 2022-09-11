import os
import torch
import numpy as np 

from config import *
from dataset import dataset
from model import CaptchaModel
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def start_trainig():
    targets, image_paths = [], []
    
    for image_name in os.listdir(DATA_DIR):
        image_path = os.path.join(DATA_DIR, image_name)
        image_paths.append(image_path)
        targets.append(image_name[:-4])
        
    targets = [[c for c in s] for s in targets]
    target_characters = list(set([c for clist in targets for c in clist]))
    lbl_enc = LabelEncoder()
    lbl_enc.fit(target_characters)
    encoded_targets = [lbl_enc.transform(target) for target in targets]
    encoded_targets = np.array(encoded_targets, dtype="uint8") + 1      # "blanck" character 
    train_image_paths, test_image_paths, train_targets, test_targets = train_test_split(image_paths, encoded_targets, test_size=0.1)
    train_dataset = dataset(train_image_paths, train_targets, transformer)
    test_dataset = dataset(test_image_paths, test_targets, transformer)

    train_dataLoader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_)).to(DEVICE)
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ================================================ TRAINING MODEL ===============================================
    for epoch in range(EPOCHS):
        # ============================================ TRAINING =====================================================
        if epoch % 10 == 0:
            train_loss, train_length_acc, train_word_acc = train(model, criterion, optimizer, train_dataLoader, lbl_enc, calc_acc=True)
            test_loss, test_length_acc, test_word_acc = test(model, criterion, test_dataLoader, lbl_enc, calc_acc=True)
            print(f"Epoch: {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | Train length acc: {train_length_acc:.4f} | Train word acc: {train_word_acc:.4f}")
            print(f"Epoch: {epoch}/{EPOCHS} | Test loss: {test_loss:.4f} | Test length acc: {test_length_acc:.4f} | Test word acc: {test_word_acc:.4f}")
        else:
            train_loss = train(model, criterion, optimizer, train_dataLoader, lbl_enc)
            test_loss = test(model, criterion, test_dataLoader, lbl_enc)
            print(f"Epoch: {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
    
    train_loss, train_length_acc, train_word_acc = train(model, criterion, optimizer, train_dataLoader, lbl_enc, calc_acc=True)
    test_loss, test_length_acc, test_word_acc = test(model, criterion, test_dataLoader, lbl_enc, calc_acc=True)
    
    torch.save({
        "epoch": epoch+1,
        "model_state": model.state_dict(),
        "train_loss": train_loss.item(),
        "test_loss": test_loss.item(),
        "train_length_acc": train_length_acc,
        "test_length_acc": test_length_acc,
        "train_word_acc": train_word_acc,
        "test_word_acc": test_word_acc
    }, "../models/last_model.pt")
    print("\nModel saved successfully")

if __name__ == "__main__":    
    start_trainig() 



