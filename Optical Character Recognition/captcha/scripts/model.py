import torch
import torch.nn as nn
import torch.nn.functional as F



class CaptchaModel(nn.Module):
    def __init__(self, num_chars): # image size = (3, 300, 75)
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        
        self.linear_1 = nn.Linear(in_features=448, out_features=16)
        self.drop_1 = nn.Dropout(0.2)
        
        self.gru = nn.GRU(16, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(32*2, num_chars + 1)
        
        
    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        
        x = self.max_pool_1(F.relu(self.conv1(images)))
        x = self.max_pool_2(F.relu(self.conv2(x)))
        
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(bs, x.size(1), -1)

        x = self.drop_1(F.relu(self.linear_1(x)))        
        
        x, _ = self.gru(x)
        
        x = F.log_softmax(self.output(x), 2)
        
        return x

        
        
if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(8, 3, 30, 120)
    target = torch.randint(1, 20, (1, 5))
    x = cm(img, target)
    print(x.shape)
