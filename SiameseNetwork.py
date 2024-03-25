import torch
import torch.nn as nn
import torch.optim as optim


class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# siamese_net = SiameseCNN()
# criterion = nn.
# optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     for img1, img2, labels in train_loader:
#         optimizer.zero_grad()
#         output1, output2 = siamese_net(img1, img2)
#         loss = criterion(output1 - output2, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
