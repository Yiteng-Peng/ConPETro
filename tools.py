import torch
from torchvision import datasets, transforms
from model import *
from tenseal_model import *
from mnistm import MNISTM
import numpy as np
from tqdm import tqdm


VAL_TRAIN_RATIO = 0.1
TEST_RATIO = 0.2


def utility_function(delta_asr, acc, alpha = 0.7):
    return alpha * acc + (1 - alpha) * delta_asr


def get_mean_std(data_name):
    data_name = data_name.lower()

    if data_name == "mnist":
        mean, std = [0.1307], [0.3015]
    elif data_name == "fmnist":
        mean, std = [0.2860], [0.3205]
    elif data_name == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif data_name == "mnistm":
        mean, std = [0.4639, 0.4676, 0.4199], [0.1976, 0.1843, 0.2078]
    elif data_name == "credit":
        return None, None
    elif data_name == "bank":
        return None, None
    else:
        raise NotImplementedError(data_name)
    
    return torch.tensor(mean), torch.tensor(std)


def get_poly_degree(data_name):
    data_name = data_name.lower()
    dummy_plain_model = load_model(data_name)
    dummy_ts_model = load_ts_model(dummy_plain_model, data_name)
    return dummy_ts_model.activation_degree


# Pass in [1, 2, 3]: Select 1,2,3 these labels
# Pass in [-1, x]: Select all but x
def get_label_data(loader, labels=None, num=None):
    is_exclude_mode = len(labels) == 2 and labels[0] == -1

    x_data, y_data = None, None

    for i, (x_batch, y_batch) in enumerate(loader):
        if labels is not None:
            if is_exclude_mode:
                
                exclude_label = labels[1]
                exclude_tensor = torch.tensor(exclude_label, dtype=y_batch.dtype, device=y_batch.device)
                mask = (y_batch != exclude_tensor)
            else:
                
                labels_tensor = torch.as_tensor(labels, dtype=y_batch.dtype, device=y_batch.device)
                mask = torch.isin(y_batch, labels_tensor)
            indices = mask.nonzero(as_tuple=True)[0]
        else:
            
            indices = torch.arange(y_batch.size(0), device=y_batch.device)

        
        if indices.numel() > 0:  
            if x_data is None:
                x_data = x_batch[indices]
                y_data = y_batch[indices]
            else:
                x_data = torch.cat((x_data, x_batch[indices]))
                y_data = torch.cat((y_data, y_batch[indices]))

        
        if num is not None and x_data is not None and x_data.size(0) >= num:
            break

    if num is not None:
        return x_data[:num], y_data[:num]
    else:
        return x_data, y_data


def load_model(data_name):
    if data_name == "mnist":
        return FMNIST_Sigmoid() 
    elif data_name == "fmnist":
        return FMNIST_Sigmoid()
    elif data_name == "mnistm":
        return MNISTM_Tanh()
    elif data_name == "cifar10":
        return CIFAR10_GeLU()
    elif data_name == "credit":
        return Credit_Sigmoid()
    elif data_name == "bank":
        return Bank_Tanh()
    else:
        raise NotImplementedError(f"Not implement {data_name.upper()}")


def load_features_num(data_name):
    if data_name == "mnist":
        in_features = [1, 28, 28]
        out_features = 10
    elif data_name == "fmnist":
        in_features = [1, 28, 28]
        out_features = 10
    elif data_name == "mnistm":
        in_features = [3, 28, 28]
        out_features = 10
    elif data_name == "cifar10":
        in_features = [3, 32, 32]
        out_features = 10
    elif data_name == "credit":
        in_features = [23]
        out_features = 2
    elif data_name == "bank":
        in_features = [20]
        out_features = 2
    else:
        raise NotImplementedError(data_name)

    return in_features, out_features


def split_val_dataset(train_dataset, test_dataset=None):
    if test_dataset is None:
        total_size = len(train_dataset)
        test_size = int(TEST_RATIO * total_size)
        train_size = total_size - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size], generator=torch.manual_seed(42)
        )

    train_size = len(train_dataset)
    val_size = int(VAL_TRAIN_RATIO * train_size)
    train_size = train_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.manual_seed(42)
    )

    return train_dataset, val_dataset, test_dataset


def load_torch_data(data_name, batch_size=64, transform=None, subset_num=None, example_num=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    if data_name == "mnist":
        train_dataset = datasets.MNIST('./dataset', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('./dataset', train=False, transform=transform)      
    elif data_name == "fmnist":
        train_dataset = datasets.FashionMNIST('./dataset', train=True, download=False, transform=transform)
        test_dataset = datasets.FashionMNIST('./dataset', train=False, transform=transform)
    elif data_name == "cifar10":
        train_dataset = datasets.CIFAR10('./dataset', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10('./dataset', train=False, transform=transform)
    elif data_name == "mnistm":
        train_dataset = MNISTM('./dataset', train=True, download=False, transform=transform)
        test_dataset = MNISTM('./dataset', train=False, transform=transform)
    elif data_name == "credit":
        features = np.load('./dataset/credit_under_sampling_data.npy')
        labels = np.load('./dataset/credit_under_sampling_label.npy')
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(features), torch.Tensor(labels))
        test_dataset = None
    elif data_name == "bank":
        features = np.load('./dataset/bank_under_sampling_data.npy')
        labels = np.load('./dataset/bank_under_sampling_label.npy')
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(features), torch.Tensor(labels))
        test_dataset = None

    train_dataset, val_dataset, test_dataset = split_val_dataset(train_dataset, test_dataset)  

    if subset_num is None:
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        if len(val_dataset) < subset_num:
            print(f"Length of Validation Dataloader is less than {subset_num}, use the length {len(val_dataset)} as the subset number")
            subset_num = len(val_dataset)

        indices = list(range(subset_num))
        subset = torch.utils.data.Subset(val_dataset, indices)                
        valid_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if example_num is not None:
        example_list = []
        cur_num = 0
        for data, _ in valid_loader:
            if cur_num >= example_num:
                break
            example_list.append(data)
            cur_num += data.size(0)
        example_data = torch.cat(example_list)[:example_num]
        example_data_numpy = example_data.numpy()
        return train_loader, valid_loader, test_loader, example_data_numpy
    else:
        return train_loader, valid_loader, test_loader
    

# calculate mean and std for each dataset
def _cal_mean_std(data_name):
    data_name = data_name.lower()

    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if data_name == "mnist":
        train_dataset = datasets.MNIST('./dataset', train=True, download=False, transform=transform)      
    elif data_name == "fmnist":
        train_dataset = datasets.FashionMNIST('./dataset', train=True, download=False, transform=transform)
    elif data_name == "cifar10":
        train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
    elif data_name == "mnistm":
        train_dataset = MNISTM('./dataset', train=True, download=False, transform=transform)

    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    channel = None
    for inputs, targets in dataloader:
        channel = inputs.shape[1]
        break

    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    print('==> Computing mean and std..')
    for inputs, targets in tqdm(dataloader):
        for i in range(channel):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    
    print(f'mean: {mean}')   
    print(f'std: {std}')