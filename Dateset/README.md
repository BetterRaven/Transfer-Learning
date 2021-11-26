此数据集为凯斯西储大学轴承数据集，每个文件夹下保存的是不同马力下的数据。<br>
使用data_loader.load_data，会将数据文件<b>'all_data.mat'</b>读取为224*224的矩阵，并转为torch的dataloader，适合于resnet18的输入。
