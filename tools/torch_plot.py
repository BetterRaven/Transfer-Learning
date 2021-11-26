import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
rc('font', family='KaiTi')
rcParams['axes.unicode_minus'] = False


def plot(train, test, dtype='acc', save=False, filename=None):
    """
    画出训练和测试的对比曲线
    :param dtype: 变量类型，acc或者loss
    :param train: 训练期间的数据
    :param test: 测试期间的数据
    :param save: 是否保存图片，默认不保存
    :param filename: 如果“save” 为True，那么必须指定文件名参数
    :return:
    """
    if len(train) != len(test):
        raise ValueError("x1 and x2 must have the same shape")

    epoch = list(range(1, len(train) + 1))
    plt.plot(epoch, train, color='r', ls='-', label='train')
    plt.plot(epoch, test, color='g', ls=':', label='test')
    plt.xlabel('Epoch')
    plt.ylabel(dtype)
    plt.title(dtype)
    plt.legend(loc='best')
    if save:
        if not filename:
            raise ValueError("Filename is empty")
        plt.savefig(filename, dpi=330)
    plt.show()
