from ImagesDataset import *
from SiameseNetwork import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
directory = r'C:\Users\User\.kaggle'
print(device)
d = ImagesDataset(file_path=directory)
d.print_stats()
loader_train = DataLoader(d, batch_size=64, shuffle=True, pin_memory=True)

siamese_net = SiameseCNN()
# siamese_net.load_state_dict(torch.load('model2.pth'))
siamese_net.to(device)
criterion = ContrastiveLoss()
criterion.to(device)
optimizer = optim.Adam(siamese_net.parameters(), lr=0.00005, weight_decay=1e-6)
siamese_net.train()
ep_acc = []
ep_loss = []
for i in range(25):
    print(f"epoch {i}")
    counter = 0
    total_loss = 0
    acc = 0
    for image1, image2, label in loader_train:
        optimizer.zero_grad()
        output1, output2 = siamese_net(image1.float().to(device), image2.float().to(device))
        label = label.view(-1, 1).to(device)
        loss = criterion(output1, output2, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        counter += 1
        # euclidean_distance = 1 - F.cosine_similarity(output1, output2)

        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = euclidean_distance < 0.5
        label = label < 1
        acc += (euclidean_distance == label).sum().item() / len(label)
    ep_acc.append(acc / counter)
    ep_loss.append(total_loss / counter)
    print(f"the loss is {total_loss / counter}, accuracy is {acc / counter} in {counter}")
    torch.save(siamese_net.state_dict(), 'model3.pth')


plt.plot(range(1, 26), ep_loss, label='Train loss')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss')
plt.legend()
plt.show()
plt.close()

# Plot accuracy
plt.plot(range(1, 26), ep_acc, label='Accuracy')
plt.title(f"Siamese Network")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

siamese_net.eval()
t = SubmissionInfo(file_path=directory)
t.print_stats()
test_loader = DataLoader(t, batch_size=64, shuffle=True, pin_memory=True)
with torch.no_grad():
    counter = 0
    acc = 0
    for image1, image2, label in test_loader:
        counter += 1
        output1, output2 = siamese_net(image1.float().to(device), image2.float().to(device))
        label = label.to(device)
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = euclidean_distance < 0.5
        label = label == 0
        acc += sum(euclidean_distance == label) / len(label)
    print(f"accuracy is {acc / counter} in {counter}")

