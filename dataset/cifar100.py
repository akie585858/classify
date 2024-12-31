from mylib.datas.cifar import get_cifar100
from torch.utils.data import Dataset
from torchvision import transforms
import torch

CIFAR100_MEAN = [0.5071, 0.4866, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]

class CIFAR100(Dataset):
    def __init__(self, train=True, img_size=32, data_argument=None):

        base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN,
                                    std=CIFAR100_STD)
        ])

        if data_argument is None:
            transform = base_transform
        else:
            transform = transforms.Compose([
                data_argument,
                base_transform
            ])
         
        self.data = get_cifar100(train, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.__getitem__(index)


if __name__ == '__main__':
    data = CIFAR100()

    x_list = []
    for x, y in data:
        x_list.append(x)
    x_list = torch.stack(x_list)
    x_u = torch.mean(x_list, dim=(0, 2, 3))
    x_std = torch.std(x_list, dim=(0, 2, 3))
    print(x_u)
    print(x_std)