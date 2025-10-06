import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms as T
import torch.nn.functional as F
from tools import load_torch_data, load_model, get_mean_std
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.softmax(output, dim=1)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()


def test_epoch(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def normal_train(data_name, epochs):
    data_name = data_name.lower()

    mean, std = get_mean_std(data_name)
    if data_name == "cifar10":
        transform = T.Compose([T.RandomCrop(32, padding=4), 
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(mean, std)])
    elif data_name in ["fmnist", "mnistm"]:
        transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean, std)])
    elif data_name in ["credit", "bank"]:
        transform = None
    else:
        raise NotImplementedError(data_name) 

    normal_model = load_model(data_name).to(device)

    train_loader, valid_loader, test_loader = load_torch_data(data_name, transform=transform)
    train_loader.shuffle = True
    
    optimizer = optim.Adam(normal_model.parameters(), weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(1, epochs + 1))

    best_acc = 0
    for epoch in pbar:
        train_epoch(normal_model, train_loader, optimizer, criterion)
        
        normal_model.eval()
        acc = test_epoch(normal_model, valid_loader)
        normal_model.train()

        if acc > best_acc:
            best_acc = acc
            torch.save(normal_model.state_dict(), f'./pretrained/normal_{data_name}.pt')
        pbar.set_postfix({'Epoch': epoch, 'ValAcc': f'{acc:.2f}%', 'ValBest': f'{best_acc:.2f}%'})


def normal_test(data_name):
    data_name = data_name.lower()
    mean, std = get_mean_std(data_name)
    if data_name in ["fmnist", "mnistm", "cifar10"]:
        transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean, std)])
    elif data_name in ["credit", "bank"]:
        transform = None
    else:
        raise NotImplementedError(data_name) 

    normal_model = load_model(data_name).to(device)
    normal_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', weights_only=True))
    normal_model.eval()

    train_loader, valid_loader, test_loader = load_torch_data(data_name, transform=transform, batch_size=1)
    acc = test_epoch(normal_model, test_loader)
    print(f"{data_name} accuracy: {acc:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='FMNIST', choices=['FMNIST', 'MNISTM', 'CIFAR10', "Credit", "Bank"])
    parser.add_argument('-f', '--func', type=str, choices=['normal_train', 'normal_test'])
    parser.add_argument('-e', '--epochs', type=int, default=100)
    args = parser.parse_args()

    if args.func == "normal_train":
        normal_train(args.dataset, args.epochs)
    elif args.func == "normal_test":
        normal_test(args.dataset)
    else:
        raise NotImplementedError(f"Not implement f{args.func}")