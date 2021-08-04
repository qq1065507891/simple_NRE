import torch
import torch.nn as nn
import time

from NRE.utils import ensure_dir


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load_model(self, path):
        """
        加载模型
        :param path:
        :return:
        """
        self.load_state_dict(torch.load(path))

    def save_model(self, epoch=0, name=None):
        """
        保存模型
        :param epoch:
        :param name:
        :return:
        """
        prefix = 'checkpoints/'
        ensure_dir(prefix)
        if name is None:
            name = prefix + self.model_name + "_" + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')
        else:
            name = prefix + name + '_' + self.model_name + "_" + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')

        torch.save(self.state_dict(), name)
        return name
