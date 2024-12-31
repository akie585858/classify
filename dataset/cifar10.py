from mylib.datas.cifar import get_cifar10
from torch.utils.data import Dataset
from torchvision import transforms
import torch

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

class CIFAR10(Dataset):
    def __init__(self, train=True, data_argument=None):

        base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN,
                                    std=CIFAR10_STD)
        ])

        if data_argument is None:
            transform = base_transform
        else:
            transform = transforms.Compose([
                base_transform,
                data_argument
            ])
         
        self.data = get_cifar10(train, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.__getitem__(index)


if __name__ == '__main__':
    data = CIFAR10()

    x_list = []
    for x, y in data:
        x_list.append(x)
    x_list = torch.stack(x_list)
    x_u = torch.mean(x_list, dim=(0, 2, 3))
    x_std = torch.std(x_list, dim=(0, 2, 3))
    print(x_u)
    print(x_std)
