from scipy.io import loadmat
import os
from os.path import join
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch


root = r'E:\Raven\jupyter\Transfer Learning\Dateset\凯斯西储大学数据'
frequency = '12K'
classes = ('滚动体故障', '内圈故障', '外圈故障', '正常')
where = '风扇端'
source = '0HP'
target = '1HP'


def load_data(root, frequency, classes, hp, where, dtype='DE', batch_size=32):
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