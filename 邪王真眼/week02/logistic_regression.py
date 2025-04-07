import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from sklearn.datasets import load_breast_cancer


class LogisticRegression(nn.Module):
    def __init__(self, input_size=30):
        super(LogisticRegression, self).__init__()
        
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, 1)
        self.sigmod = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('sigmoid'))
        init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        x = self.sigmod(x)
        return x

def train_and_val(train_X, val_X , train_y, val_y, lr):
    # model
    model = LogisticRegression(input_size=30)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    max_correct = 0
    best_model = None
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()

        optimizer.step()

        predicted = (outputs > 0.5).int()
        correct = predicted.eq(train_y.int()).sum().item()

        print(f'\r  TRAIN: Epoch [{epoch + 1}/10000, Loss: {loss:.4f}, Accuracy: {100.*correct/train_y.size(0):.2f}%]', end='', flush=True)

        # val
        outputs = model(val_X)
        loss = criterion(outputs, val_y)
        predicted = (outputs > 0.5).int()
        correct = predicted.eq(val_y.int()).sum().item()

        if correct > max_correct:
            max_correct = correct
            best_model = copy.deepcopy(model)

    # save model
    save_path = "./LogisticRegression_breast_cancer.pth"
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(best_model.state_dict(), save_path)

    print(f'\n  VAL:   Accuracy: {100.*max_correct/val_y.size(0):.2f}%]')

def main():
    # hyperparameter
    train_precent = 0.7
    lr_list = [0.01]

    # dataset
    X, y = load_breast_cancer(return_X_y=True)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)#.clamp(max=1)

    num_data = len(X)
    num_train = int(np.floor(train_precent * num_data))

    train_X, val_X = X[:num_train], X[num_train:]
    train_y, val_y = y[:num_train].unsqueeze(1), y[num_train:].unsqueeze(1)

    # train
    for lr in lr_list:
        print(f"lr: {lr}")
        train_and_val(train_X, val_X , train_y, val_y, lr)

if __name__ == "__main__":
    main()
