import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0ForCustomTask(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0ForCustomTask, self).__init__()

        # EfficientNet-B0 with pre-trained weights
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Freeze pre-trained layers
        self.model.eval()
        self.model.classifier =nn.Identity()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.output = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.output(x)
        x = self.softmax(x)

        return x
