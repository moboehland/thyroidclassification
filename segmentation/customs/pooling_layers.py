from torch import nn
from customs.activation_functions import Mish


class ConvPool(nn.Module):

    def __init__(self, ch_in, act_fun, normalization):
        """

        :param ch_in:
        :param act_fun:
        :param normalization:
        """

        super().__init__()
        self.conv_pool = list()

        self.conv_pool.append(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1, bias=True))

        if act_fun == 'relu':
            self.conv_pool.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv_pool.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv_pool.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv_pool.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        if normalization == 'bn':
            self.conv_pool.append(nn.BatchNorm2d(ch_in))
        elif normalization == 'gn':
            self.conv_pool.append(nn.GroupNorm(num_groups=8, num_channels=ch_in))
        elif normalization == 'in':
            self.conv_pool.append(nn.InstanceNorm2d(num_features=ch_in))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv_pool = nn.Sequential(*self.conv_pool)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv_pool)):
            x = self.conv_pool[i](x)

        return x