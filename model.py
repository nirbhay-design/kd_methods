import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TeacherResNet(nn.Module):
    def __init__(self, num_classes=10, arch='resnet18'):
        super(TeacherResNet, self).__init__()
        
        if arch == 'resnet50':
            base_model = models.resnet50(weights=None)
        else:
            base_model = models.resnet18(weights=None)
            
        # Modify for CIFAR-10
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
        # Extract features up to the final pooling layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        rep = x.view(x.size(0), -1)
        logits = self.fc(rep)
        return logits, rep

class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 pools, spatial dims go from 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        rep = x.view(x.size(0), -1)
        logits = self.fc(rep)
        return logits, rep