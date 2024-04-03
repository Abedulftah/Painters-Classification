import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.pretrained_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward_once(self, x):
        x = self.pretrained_model(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        # Having a margin indicates that dissimilar pairs that are beyond this margin will not contribute to the loss.
        self.margin = margin

    def forward(self, output1, output2, label):
        # label == 1 if they are not the same artist else 0.
        # pairwise_distance function does the normalization for us, with default p=2.
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # torch.mean is used to take the mean of all the individual losses computed for the pairs in the batch. This
        # ensures that the loss value is normalized with respect to the batch size.
        loss_contrastive = torch.mean((1-label) * euclidean_distance +
                                      label * torch.clamp(self.margin - euclidean_distance, min=0.0))
        return loss_contrastive

