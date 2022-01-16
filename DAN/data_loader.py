from torchvision import datasets, transforms
import torch
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from os.path import join


def load_data(root_path, dir, batch_size):
    data = scio.loadmat(join(root_path, dir, "all_data.mat"))
    label = scio.loadmat(join(root_path, dir, "all_label.mat"))
    label = label.get('label')
    data = data.get('feature')
    permutation = np.random.permutation(data.shape[0])
    # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
    data = data[permutation]                             # 得到打乱后数据data
    label = label[permutation]                             # 得到打乱后数据label
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    label = torch.utils.data.DataLoader(label, batch_size=batch_size, shuffle=False, drop_last=True)

    return data, label


def load_training(root_path, dir, batch_size, kwargs):
    #多个图片操作的串联
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    #数据加载器
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader


def num_to_string(num):
    numbers = {
        0: "back_pack",
        1: "bike",
        2: "bike_helmet",
        3: "bookcase",
        4: "bottle",
        5: "calculator",
        6: "desk_chair",
        7: "desk_lamp",
        8: "desktop_computer",
        9: "file_cabinet",
        10: "headphones",
        11: "keyboard",
        12: "laptop_computer",
        13: "letter_tray",
        14: "mobile_phone",
        15: "monitor",
        16: "mouse",
        17: "mug",
        18: "paper_notebook",
        19: "pen",
        20: "phone",
        21: "printer",
        22: "projector",
        23: "punchers",
        24: "ring_binder",
        25: "ruler",
        26: "scissors",
        27: "speaker",
        28: "stapler",
        29: "tape_dispenser",
        30: "trash_can"
        }
    return numbers.get(num)


def plot_image(img, label, name):
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        imge=img[i]
       
        # img = img.numpy()  # FloatTensor转为ndarray
        imge = np.array(imge)
        
        imge= np.transpose(imge, (1, 2, 0))  # 把channel那一维放到最后
       
        # img=img.view(224,224,-1)
        plt.imshow(imge)
        
        plt.title("{}: {}".format(name, num_to_string(label[i].item())))
        plt.xticks([])
        plt.yticks([])
    plt.show()