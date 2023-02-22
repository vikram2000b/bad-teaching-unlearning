from torchvision.datasets import CIFAR100
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.coarse_map = {
            0:[4, 30, 55, 72, 95],
            1:[1, 32, 67, 73, 91],
            2:[54, 62, 70, 82, 92],
            3:[9, 10, 16, 28, 61],
            4:[0, 51, 53, 57, 83],
            5:[22, 39, 40, 86, 87],
            6:[5, 20, 25, 84, 94],
            7:[6, 7, 14, 18, 24],
            8:[3, 42, 43, 88, 97],
            9:[12, 17, 37, 68, 76],
            10:[23, 33, 49, 60, 71],
            11:[15, 19, 21, 31, 38],
            12:[34, 63, 64, 66, 75],
            13:[26, 45, 77, 79, 99],
            14:[2, 11, 35, 46, 98],
            15:[27, 29, 44, 78, 93],
            16:[36, 50, 65, 74, 80],
            17:[47, 52, 56, 59, 96],
            18:[8, 13, 48, 58, 90],
            19:[41, 69, 81, 85, 89]
        }
        
    #def __len__(self):
    #    len(self.main_dataset)
        
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y
    
    
class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y