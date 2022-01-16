import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 屏蔽GPU
os.sys.path.append(r'F:\work\jupyter\Transfer Learning')
from tools import mmd
from tools import torch_plot
from tools import load
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 文件路径等
root = r'F:\凯斯西储大学数据'
frequency = '12K'
classes = ('滚动体故障', '内圈故障', '外圈故障', '正常')
where = '风扇端'
source = '0HP'
target = '1HP'
batch_size = 32