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
    return:返回的是224*224的矩阵，resnet18要求的数据
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

def load_cwru(root, frequency, classes, hp, where, dtype='DE', batch_size=32):
    """
    功能：加载指定马力的数据
    参数：root:数据集的根目录
         frequency:使用什么平频率的数据，只能是12K或者48k
         classes:每个子类的名称，默认4类（滚动体故障, 内圈故障, 外圈故障, 正常）
         hp:负载值，只有0HP、1HP、2HP、3HP
         where:使用风扇端还是驱动端
         dtype:使用何种数据，只有DE、FE、BA
         batch_size:批大小，默认32
    返回：x_data:数据，类型为DataLoader，每个样本的大小为64*64
         label:标签，类型为DataLoader
    """
    x_data = None
    label = None
    for index, item in enumerate(classes):
        # 生成文件绝对路径
        mat_path = join(root, frequency, item, where, hp)
        files = os.listdir(mat_path)
        
        x_temp = None
        for file in files:
            mat = loadmat(join(mat_path, file))
        
            # 读取数据
            temp = file.split('\\')[-1].split('.')[0]  # 获取文件名
            if eval(temp) < 100:    # 仅用于处理97.mat -- 99.mat
                temp = '0' + temp
            key = 'X' + temp + '_' + dtype + '_time'
            val = mat[key]
            
            # 将数据处理成64*64
            for i in range(1, int(len(val))):
                temp = val[(i - 1) * 4096 : i * 4096]
                if len(temp) < 4096:
                    break
                # 标准化
                mm = MinMaxScaler()
                x_std = mm.fit_transform(temp.reshape((64, 64)))
                if x_temp is not None:
                    x_std = np.expand_dims(x_std, axis=0)
                    x_temp = np.append(x_temp, x_std, axis=0)
                else:
                    x_temp = np.expand_dims(x_std, axis=0)
                    
        # 读完一类了，添加标签
        if label is not None:
            label = np.append(label, np.ones(x_temp.shape[0]) * index)
        else:
            label = np.ones(x_temp.shape[0]) * index
        
        # 拼合数据
        if x_data is not None:
            x_data = np.append(x_data, x_temp, axis=0)
        else:
            x_data = x_temp
            
    # 打乱数据
    permutation = np.random.permutation(x_data.shape[0])
    # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
    x_data = x_data[permutation]
    label = label[permutation]
    
    # 转化成DataLoader
    x_data = torch.tensor(x_data)
    x_data = x_data.to(torch.float32)
    label = torch.tensor(label)
    label = label.to(torch.long)
    
    x_data = torch.unsqueeze(x_data, dim=1)    # 添加一个维度，通道数
    x_data = DataLoader(x_data, batch_size=batch_size, shuffle=False)
    label = DataLoader(label, batch_size=batch_size, shuffle=False)
    return x_data, label

def load_dds(root, classes, dtype='AllYouData6', batch_size=32, test_size=0.1):
    """
    功能：加载DDS数据集
    参数：root:数据集的根目录
         classes:每个子类的名称，默认4类（滚动体故障, 内圈故障, 外圈故障, 正常）
         dtype:使用何种数据，AllYouData6
         batch_size:批大小，默认32
         test_size:划分测试集的大小、
    返回：x_train:训练数据，类型为DataLoader，每个样本的大小为64*64
         x_test:测试数据，类型为DataLoader
         y_train:训练标签
         y_test:测试标签
    """
    x_data = None
    label = None
    for index, item in enumerate(classes):
        x_temp = None
        # 生成文件绝对路径
        mat_path = join(root, item)
        mat = loadmat(mat_path)

        # 读取数据
        key = dtype
        val = mat[key][0][0]
        print(val.shape)
        
        # 将数据处理成64*64
        for i in range(1, int(len(val))):
            temp = val[(i - 1) * 4096 : i * 4096]
            if len(temp) < 4096:
                break
            # 标准化
            mm = MinMaxScaler()
            x_std = mm.fit_transform(temp.reshape((64, 64)))
            if x_temp is not None:
                x_std = np.expand_dims(x_std, axis=0)
                x_temp = np.append(x_temp, x_std, axis=0)
            else:
                x_temp = np.expand_dims(x_std, axis=0)
                    
        # 读完一类了，添加标签
        if label is not None:
            label = np.append(label, np.ones(x_temp.shape[0]) * index)
        else:
            label = np.ones(x_temp.shape[0]) * index
        
        # 拼合数据
        if x_data is not None:
            x_data = np.append(x_data, x_temp, axis=0)
        else:
            x_data = x_temp
            
    # 打乱数据
    permutation = np.random.permutation(x_data.shape[0])
    # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
    x_data = x_data[permutation]
    label = label[permutation]
    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data, label, test_size=test_size)
    
    
    # 转化成DataLoader
    x_train = torch.tensor(x_train)
    x_train = x_train.to(torch.float32)
    x_train = torch.unsqueeze(x_train, dim=1)    # 添加一个维度，通道数 
    
    x_test = torch.tensor(x_test)
    x_test = x_test.to(torch.float32)
    x_test = torch.unsqueeze(x_test, dim=1)      # 添加一个维度，通道数 
    
    y_train = torch.tensor(y_train)
    y_train = y_train.to(torch.long)
    
    y_test = torch.tensor(y_test)
    y_test = y_test.to(torch.long)

    x_train = DataLoader(x_train, batch_size=batch_size, shuffle=False)
    y_train = DataLoader(y_train, batch_size=batch_size, shuffle=False)
    if len(x_test) < batch_size:
        x_test = DataLoader(x_test, batch_size=len(x_test), shuffle=False)
        y_test = DataLoader(y_test, batch_size=len(y_test), shuffle=False)
    else:
        x_test = DataLoader(x_test, batch_size=batch_size, shuffle=False)
        y_test = DataLoader(y_test, batch_size=batch_size, shuffle=False)
    return x_train, x_test, y_train, y_test
