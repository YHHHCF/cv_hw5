import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, mode):
        path = '../data/nist36_' + mode + '.mat'
        data = scipy.io.loadmat(path)
        self.imgs = torch.tensor(data[mode + '_data'], dtype=torch.float32)
        self.labels = torch.tensor(data[mode + '_labels'], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img.reshape((1, 32, 32))
        img = torch.cat((img, img, img), dim=0)
        label = self.labels[index]
        return img, label


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class net(nn.Module):
    def __init__(self, num_classes):
        super(net, self).__init__()
        self.conv1 = conv(3, 16)
        self.conv2 = conv(16, 16)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.flatten = Flatten()

        self.linear1 = nn.Linear(4096, 512)
        self.linear2 = nn.Linear(512, num_classes)

        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        return x


def plotline(train_data, val_data, xlabel, ylabel, title):
    plt.plot(val_data[0], val_data[1], 'r', label='val')
    plt.plot(train_data[0], train_data[1], 'b', label='train')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()


def save_ckpt(model, loss, acc):
    path = './../results/model.t7'

    torch.save({
        'loss': loss,
        'acc': acc,
        'params': model.state_dict(),
    }, path)
    return path


def load_ckpt(path):
    model = net(36)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['params'])

    print("Model has loss {} and acc {}:".format(ckpt['loss'], ckpt['acc']))
    return model


def train(model, optim, train_loader, val_loader, epoch):
    print("going to train {} epochs".format(epoch))
    global best_acc

    train_acc_list = [[], []]
    val_acc_list = [[], []]
    train_loss_list = [[], []]
    val_loss_list = [[], []]

    model.train()
    for e in range(epoch):
        total_loss = 0
        total_acc = 0
        cnt = 0
        for batch_num, (imgs, labels) in enumerate(train_loader):
            out = model(imgs)
            loss = criterion(out, torch.argmax(labels, 1))
            loss.backward()
            optim.step()

            pred = torch.argmax(out, 1)
            labels = torch.argmax(labels, 1)
            acc = torch.sum(labels == pred).double() / len(labels)
            total_loss += loss.item()
            total_acc += acc.item()
            cnt += 1

        print("Epoch {} train has loss {} and acc {}".format(e, round((total_loss / cnt), 4), total_acc / cnt))
        if e % 10 == 0:
            val_loss, val_acc = val(model, val_loader)
            print("validation has loss {} and acc {}".format(round(val_loss, 4), round(val_acc, 4)))
            if val_acc > best_acc:
                best_acc = val_acc
                save_ckpt(model, val_loss, val_acc)
                print("A new model is saved!")

            val_acc_list[0].append(e)
            val_acc_list[1].append(val_acc)
            val_loss_list[0].append(e)
            val_loss_list[1].append(val_loss)

        train_loss = total_loss / cnt
        train_acc = total_acc / cnt

        train_acc_list[0].append(e)
        train_acc_list[1].append(train_acc)
        train_loss_list[0].append(e)
        train_loss_list[1].append(train_loss)

    plotline(train_acc_list, val_acc_list, 'epoch', 'acc', 'train/val acc')
    plotline(train_loss_list, val_loss_list, 'epoch', 'loss', 'train/val loss')


def val(model, val_loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    cnt = 0

    for batch_num, (imgs, labels) in enumerate(val_loader):
        out = model(imgs)
        labels = torch.argmax(labels, 1)
        loss = criterion(out, labels)
        pred = torch.argmax(out, 1)
        acc = torch.sum(labels == pred).double() / len(labels)

        total_loss += loss.item()
        total_acc += acc.item()
        cnt += 1

    val_loss = total_loss / cnt
    val_acc = total_acc / cnt

    model.train()
    return val_loss, val_acc


if __name__ == '__main__':
    best_acc = 0

    train_dataset = ImageDataset('train')
    val_dataset = ImageDataset('valid')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    model = net(36)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    train(model, optimizer, train_loader, val_loader, 200)
