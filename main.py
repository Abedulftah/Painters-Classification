from ImagesDataset import *
from SiameseNetwork import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
directory = r'C:\Users\User\.kaggle'
print(device)
train_dataset = ImagesDataset(file_path=directory)
train_dataset.print_stats()
loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_dataset = SubmissionInfo(file_path=directory)
test_dataset.print_stats()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True)

siamese_net = SiameseCNN()
# siamese_net.load_state_dict(torch.load('model2.pth'))
siamese_net.to(device)
criterion = ContrastiveLoss()
criterion.to(device)
optimizer = optim.Adam(siamese_net.parameters(), lr=0.00005, weight_decay=1e-6)

ep_acc_train, ep_loss_train, ep_acc_test, ep_loss_test = [], [], [], []

for i in range(25):
    print(f"epoch {i}")
    counter = 0
    total_loss = 0
    acc = 0
    siamese_net.train()
    for image1, image2, label in loader_train:
        optimizer.zero_grad()
        output1, output2 = siamese_net(image1.float().to(device), image2.float().to(device))
        label = label.view(-1, 1).to(device)
        loss = criterion(output1, output2, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        counter += 1

        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = euclidean_distance < 0.5
        label = label < 1
        acc += (euclidean_distance == label).sum().item() / len(label)
    ep_acc_train.append(acc / counter)
    ep_loss_train.append(total_loss / counter)
    print(f"the train loss is {total_loss / counter}, train accuracy is {acc / counter} in {counter}")
    torch.save(siamese_net.state_dict(), 'model.pth')
    with torch.no_grad():
        siamese_net.eval()
        counter = 0
        loss = 0
        acc = 0
        for image1, image2, label in test_loader:
            counter += 1
            output1, output2 = siamese_net(image1.float().to(device), image2.float().to(device))
            label = label.view(-1, 1).to(device)
            loss += criterion(output1, output2, label).item()
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            euclidean_distance = euclidean_distance < 0.5
            label = label == 0
            acc += (euclidean_distance == label).sum().item() / len(label)
        print(f"test loss is {loss / counter}, test accuracy is {acc / counter} in {counter}")
        ep_acc_test.append(acc / counter)
        ep_loss_test.append(loss / counter)


plt.plot(range(1, 26), ep_loss_train, label='Train loss')
plt.plot(range(1, 26), ep_loss_test, label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss')
plt.legend()
plt.show()
plt.close()

# Plot accuracy
plt.plot(range(1, 26), ep_acc_train, label='Accuracy train')
plt.plot(range(1, 26), ep_acc_test, label='Accuracy test')
plt.title(f"Siamese Network")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

