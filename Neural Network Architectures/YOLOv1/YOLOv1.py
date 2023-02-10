import torch
import torch.nn as nn



class YOLOv1(nn.Module):
    def __init__(self, B, num_classes):
        super(YOLOv1, self).__init__()
        self.out_features = 5 * B + num_classes
        model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        self.extractor = nn.Sequential(*list(model.children()))[:-3]
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dense = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=self.out_features * 7*7)
        )
        
    def forward(self, x):
        x = self.extractor(x)
        x = self.layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = x.reshape(-1, 7, 7, self.out_features)
        return x
