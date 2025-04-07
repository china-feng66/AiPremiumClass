import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def main():
    # hyperparameter
    lr = 0.01
    epochs = 10
    bs = 512

    dataset_path = "./邪王真眼/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.KMNIST(root=dataset_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.KMNIST(root=dataset_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # model
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    max_correct = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        batch = 1

        print(f"\nEpoch {epoch+1}/{epochs}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            print(f'\rBatch [{batch + 1}/{len(train_loader)+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%]', end='', flush=True)
            batch += 1

        # val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'\n  VAL:   Accuracy: {100.*correct/total:.2f}%]')
        if correct > max_correct:
            max_correct = correct
            best_model = copy.deepcopy(model)
            print(f'  VAL:   best model!')

        model.train()

    # save model
    save_path = f"./邪王真眼/week03/LogisticRegression_KMNIST_acc_{100.*max_correct/total:.2f}.pth"
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(best_model.state_dict(), save_path)

if __name__ == "__main__":
    main()
