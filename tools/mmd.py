import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # batch_size = int(source.size()[0])
    # kernels = guassian_kernel(source, target,
    #                           kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # XX = kernels[:batch_size, :batch_size]
    # YY = kernels[batch_size:, batch_size:]
    # XY = kernels[:batch_size, batch_size:]
    # YX = kernels[batch_size:, :batch_size]
    # loss = torch.mean(XX + YY - XY -YX)
    # return loss

    source_num = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    target_num = int(target.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = torch.mean(kernels[:source_num, :source_num])
    YY = torch.mean(kernels[source_num:, source_num:])
    XY = torch.mean(kernels[:target_num, source_num:])
    YX = torch.mean(kernels[source_num:, :target_num])
    loss = XX + YY -XY - YX
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

# def multiple_mmd(feature, y_true, num_classes):
#     # 获取每一类数据的下标
#     class1 = []
#     class2 = []
#     class3 = []
#     class4 = []
#     for index, item in enumerate(y_true):
#         if item == 0:
#             class1.append(index)
#         elif item == 1:
#             class2.append(index)
#         elif item == 2:
#             class3.append(index)
#         else:
#             class4.append(index)
            
#     # 将每一类特征分开
#     loss = 0
#     feature_mmd = [feature[class1], feature[class2], feature[class3], feature[class4]]
    
#     # 计算MMD距离
#     for i in range(len(feature_mmd)):
#         for j in range(i+1, len(feature_mmd)):
#             loss += mmd_rbf_noaccelerate(feature_mmd[i], feature_mmd[j])
            
#     return loss

def multiple_mmd(feature, y_pred, y_true, num_classes=8):
    # 获取每一类数据的下标
    classes = []
    for i in range(num_classes):
        classes.append([])

    for index, item in enumerate(y_true):
        classes[item].append(index)
            
    # 将每一类特征分开
    loss = 0
    losses = []
    feature_mmd = []

    for item in classes:
        feature_mmd.append(feature[item])
    
    # 计算MMD距离
    for i in range(len(feature_mmd)):
        for j in range(i+1, len(feature_mmd)):
            loss += mmd_rbf_noaccelerate(feature_mmd[i], feature_mmd[j])
    
    # ---------------计算每一类的分类准确率---------------

            
    return loss