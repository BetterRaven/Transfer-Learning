import torch
import numpy as np
import scipy.io as scio
from os.path import join
import random


def load_data(root_path, sub_path, batch_size):
    """
    返回resnet18需要的数据尺寸：224x224
    """
    data = scio.loadmat(join(root_path, sub_path, "all_data.mat"))
    label = scio.loadmat(join(root_path, sub_path, "all_label.mat"))
    label = label.get('label')
    data = data.get('feature')
    permutation = np.random.permutation(data.shape[0])
    # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
    data = data[permutation]                             # 得到打乱后数据data
    label = label[permutation]                             # 得到打乱后数据label
    
    # 转为tensor
    data = torch.tensor(data)
    
    temp = torch.zeros((1000, 224, 224))
    for i in range(len(data)):
        temp[i] = data[i].repeat(7, 7)
    
    data = temp
    label = torch.tensor(label)
    data = data.to(torch.float32)
    
    data = torch.unsqueeze(data, dim=1)    # 添加一个维度，通道数
    label = torch.argmax(label, dim=1)     # 由于torch的损失函数计算时不能使用one-hot编码，这里将one-hot改了
    
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    label = torch.utils.data.DataLoader(label, batch_size=batch_size, shuffle=False, drop_last=True)

    return data, label


def partial_loader(root_path, sub_path, num_target_classes=4, batch_size=64, shuffle=True):
    """
    根据给定的类别数量，加载部分迁移的数据
    root_path:数据集的根目录
    sub_path:指定目标域
    num_target_classes:指定要加载的故障类型的类别数量
    """
    data = scio.loadmat(join(root_path, sub_path, "all_data.mat"))
    label = scio.loadmat(join(root_path, sub_path, "all_label.mat"))

    data = data.get('feature')
    label = label.get('label')
    
    # 随机选择一些类别
    if shuffle:
        classes_index = []  
        while len(classes_index) != num_target_classes:
            temp = random.randint(0, 9)
            if temp not in classes_index:
                classes_index.append(temp)

    label = np.argmax(label, axis=1)
    data_index = []
    
    # 根据classes_index选择相应的数据和标签
    for index, item in enumerate(label):
        if item in classes_index:
            data_index.append(index)
    temp = data[data_index]
    label = label[data_index]

    # 打乱数据
    permutation = np.random.permutation(temp.shape[0])
    temp = temp[permutation]                           
    label = label[permutation]  

    # 构建数据
    data = None
    temp = torch.tensor(temp)
    data = torch.zeros((temp.shape[0], 224, 224))
    for i in range(len(data)):
        data[i] = temp[i].repeat(7, 7)

    # 转为tensor
    label = torch.tensor(label)
    data = data.to(torch.float32)
    data = torch.unsqueeze(data, dim=1)    # 添加一个维度，通道数
    
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    label = torch.utils.data.DataLoader(label, batch_size=batch_size, shuffle=False, drop_last=True)
    return data, label
