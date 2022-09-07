import os
import torch 
import pickle

from config import *
from PIL import Image
from model import CaptchaModel


targets, image_paths = [], []

for image_name in os.listdir(DATA_DIR):
    image_path = os.path.join(DATA_DIR, image_name)
    image_paths.append(image_path)
    targets.append(image_name[:-4])

with open("../encoder/encoder.pickle", "rb") as f:
    lbl_enc = pickle.load(f)


batch_images, batch_targets = [], []

for i in range(20):
    target = targets[i]
    image_path = image_paths[i]
    
    image = Image.open(image_path).convert("RGB").resize(size=(120, 30))
    image = np.array(image) / 255.0
    image = torch.FloatTensor(image)[None]
    image = image.permute(0, 3, 1, 2)
    
    batch_images.append(image)
    batch_targets.append(target)


batch_images = torch.cat(batch_images, dim=0).to(DEVICE)

model = CaptchaModel(num_chars=19).to(DEVICE)
model.load_state_dict(torch.load("../models/last_model.pt")['model_state'])

pred = model(batch_images)

plt_show(pred.size(0), batch_images, pred, targets, lbl_enc)


