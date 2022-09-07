import torch
import numpy as np
import matplotlib.pyplot as plt 

from tqdm import tqdm
from torchvision import transforms


DATA_DIR = "../data/captcha_images_v2"
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 30
EPOCHS = 200
LR = 0.001
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), # (h, w)
    transforms.ToTensor(),
])

def remove_duplicates(t):
    fin, ad = "", 0

    for i in range(len(t)):
        char = t[i]
        if len(fin) == 0:
            if char != "":
                fin += char
        else:
            if char == "":
                ad = 1
            else: #fin[-1] == char:
                if ad:
                    fin += char
                    ad = 0
    return fin


def decode_predictions(out, encoder):
    # out = out.permute(1, 0, 2)
    # out = torch.softmax(out, 2)
    out = torch.argmax(out, -1)
    out = out.detach().cpu().numpy() - 1
    temp = []
    for i in out:
        if i == -1:
            temp.append("")
        else:
            i = encoder.inverse_transform([i])[0]
            temp.append(i)
    return remove_duplicates(temp)


def plt_show(bs, images, pred, text_targets, lbl_enc):
    predicted_targets = []
    
    for i in range(bs):
        predicted = decode_predictions(pred[i], lbl_enc)
        predicted_targets.append(predicted)
    
    rows, columns = 4, 5
    fig, axes = plt.subplots(figsize=(10,6), nrows=rows, ncols=columns, sharey=True)
    
    for i in range(bs):
        img = images[i].permute(1, 2, 0).cpu()
        ax = axes.flat[i]
        ax.imshow(img)
        ax.set_title(f"Target: {text_targets[i]}", fontsize=10)
        ax.set_xlabel(f"Predicted: {predicted_targets[i]}", fontsize=10)
        
    plt.subplots_adjust(top=0.85, wspace=0.15)
    plt.show()


def metrics_(bs, pred, targets, lbl_enc):
    total, word_correct, length_correct = 0, 0, 0
    
    for i in range(bs):
        predicted = decode_predictions(pred[i], lbl_enc)
        groundTruth = lbl_enc.inverse_transform(targets[i].cpu()-1)
        groundTruth = "".join(x for x in groundTruth)
        if len(predicted) == len(groundTruth):
            length_correct += 1
            if predicted == groundTruth:
                word_correct += 1
        total += 1
    len_corr = length_correct / total
    word_corr = word_correct / total
    
    return len_corr, word_corr


def train(model, criterion, optimizer, dataLoader, lbl_enc, calc_acc=None):
    model.train()
    batch_loss, batch_size = 0, 0
    len_correct, word_correct = [], [] 
    
    for images, targets in tqdm(dataLoader, position=0, total=len(dataLoader)):
        optimizer.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        pred = model(images)
        pred = pred.permute(1, 0, 2) # pred.shape ==> (num_sequence, batch_size, num_classes)
        b_size = pred.shape[1]
        batch_size += b_size
        input_lengths = torch.IntTensor(b_size).fill_(27)
        target_lengths = torch.IntTensor([len(t) for t in targets])
        loss = criterion(pred, targets, input_lengths, target_lengths)
        loss.backward()
        batch_loss += loss 
        optimizer.step()
        
        if calc_acc:
            len_corr, word_corr = metrics_(b_size, pred.permute(1, 0, 2), targets, lbl_enc)
            len_correct.append(len_corr)
            word_correct.append(word_corr)
            
    final_loss = batch_loss / batch_size
    
    if calc_acc:
        return final_loss, np.mean(len_correct), np.mean(word_correct)
    
    return final_loss



def test(model, criterion, dataLoader, lbl_enc, calc_acc=None):
    model.eval()
    batch_loss, batch_size = 0, 0
    len_correct, word_correct = [], [] 
    
    for images, targets in tqdm(dataLoader, position=0, total=len(dataLoader)):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        pred = model(images)
        pred = pred.permute(1, 0, 2) # pred.shape ==> (num_sequence, batch_size, num_classes)
        b_size = pred.shape[1]
        batch_size += b_size
        input_lengths = torch.IntTensor(b_size).fill_(27)
        target_lengths = torch.IntTensor([len(t) for t in targets])
        batch_loss += criterion(pred, targets, input_lengths, target_lengths)
        
        if calc_acc:
            len_corr, word_corr = metrics_(b_size, pred.permute(1, 0, 2), targets, lbl_enc)
            len_correct.append(len_corr)
            word_correct.append(word_corr)
    final_loss = batch_loss / batch_size
    
    if calc_acc:
        return final_loss, np.mean(len_correct), np.mean(word_correct)
    
    return final_loss



def plt_show1(bs, images, predicted_targets, text_targets):
    rows, columns = (bs//5)+1, bs
    fig = plt.figure(figsize=(5, 5))
    
    for i in range(bs):
        img = images[i].permute(1, 2, 0).cpu()
        fig.add_subplot(rows, columns, (i+1))
        plt.imshow(img)
        plt.title(f"Target: {text_targets[i]}")
        plt.xlabel(f"Predicted: {predicted_targets[i]}")
    plt.show()