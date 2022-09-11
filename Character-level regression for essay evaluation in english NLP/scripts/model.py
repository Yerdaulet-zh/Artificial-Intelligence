from torch import nn


class NeuralNet(nn.Module): # (N, C, L) --> N - Batch size | C - Num of chanels | L - length of signal sequence
    def __init__(self, in_channels, num_classes):
        super(NeuralNet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channels, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64*2, kernel_size=2, stride=1),
            nn.BatchNorm1d(64*2),
            nn.LeakyReLU(64*2)
        )
        self.fc = nn.Linear(in_features=91, out_features=48)
        self.relu = nn.LeakyReLU(48)
        self.gru = nn.GRU(48, 48, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.fc1 = nn.Linear(in_features=128*48*2, out_features=num_classes)
        
    
    def forward(self, x):
        bs = x.size(0)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.fc(x)
        x = self.relu(x)
        x, _ = self.gru(x)
        x = x.reshape(bs, -1)
        x = self.fc1(x)
        
        return x