import torch
import torch.nn as nn
from customs.activation_functions import Mish
from customs.pooling_layers import ConvPool


class ConvBlock(nn.Module):
    """ Basic convolutional block of a U-net. """

    def __init__(self, ch_in, ch_out, act_fun, normalization):
        """

        :param ch_in:
        :param ch_out:
        :param act_fun:
        :param normalization:
        """

        super().__init__()
        self.conv = list()

        # 1st convolution
        self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 1st activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 1st normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm2d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm2d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        # 2nd convolution
        self.conv.append(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 2nd activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 2nd normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm2d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm2d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv)):
            x = self.conv[i](x)

        return x


class TranspConvBlock(nn.Module):
    """ Upsampling block of a unet (with transposed convolutions). """

    def __init__(self, ch_in, ch_out, normalization):
        """

        :param ch_in:
        :param ch_out:
        :param normalization:
        """
        super().__init__()

        self.up = nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2))
        if normalization == 'bn':
            self.norm = nn.BatchNorm2d(ch_out)
        elif normalization == 'gn':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=ch_out)
        elif normalization == 'in':
            self.norm = nn.InstanceNorm2d(num_features=ch_out)
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (upsampled feature maps).
        """
        x = self.up(x)
        x = self.norm(x)

        return x


class UNet(nn.Module):
    """ UNet for distance transform or adapted boarder """

    def __init__(self, hparams):
        """

        :param ch_in:
        :param pool_method:
        :param act_fun:
        :param normalization:
        :param filters:
        """

        super().__init__()
        self.ch_in = hparams.ch_in
        self.pool_method = hparams.pool_method
        self.act_fun = hparams.activation_func
        self.normalization = hparams.normalization_func
        self.filters = [hparams.filters_first_conv, hparams.filters_last_conv]
        # ch_in=1, pool_method='conv', act_fun='relu', normalization='bn', filters=(64, 1024)): Standardparameter Tim

        # Encoder
        self.encoderConv = nn.ModuleList()

        if self.pool_method == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pool_method == 'conv':
            self.pooling = nn.ModuleList()

        # First encoder block
        n_featuremaps = self.filters[0]
        self.encoderConv.append(ConvBlock(ch_in=self.ch_in,
                                          ch_out=n_featuremaps,
                                          act_fun=self.act_fun,
                                          normalization=self.normalization))
        if self.pool_method == 'conv':
            self.pooling.append(ConvPool(ch_in=n_featuremaps, act_fun=self.act_fun, normalization=self.normalization))

        # Remaining encoder blocks
        while n_featuremaps < self.filters[1]:

            self.encoderConv.append(ConvBlock(ch_in=n_featuremaps,
                                              ch_out=(n_featuremaps*2),
                                              act_fun=self.act_fun,
                                              normalization=self.normalization))

            if n_featuremaps * 2 < self.filters[1] and self.pool_method == 'conv':
                self.pooling.append(ConvPool(ch_in=n_featuremaps*2, act_fun=self.act_fun, normalization=self.normalization))

            n_featuremaps *= 2

        # Decoder 1 (borders, seeds) and Decoder 2 (cells)
        self.decoder1Upconv = nn.ModuleList()
        self.decoder1Conv = nn.ModuleList()
        self.decoder2Upconv = nn.ModuleList()
        self.decoder2Conv = nn.ModuleList()

        while n_featuremaps > self.filters[0]:
            self.decoder1Upconv.append(TranspConvBlock(ch_in=n_featuremaps,
                                                       ch_out=(n_featuremaps // 2),
                                                       normalization=self.normalization))
            self.decoder1Conv.append(ConvBlock(ch_in=n_featuremaps,
                                               ch_out=(n_featuremaps // 2),
                                               act_fun=self.act_fun,
                                               normalization=self.normalization))
            self.decoder2Upconv.append(TranspConvBlock(ch_in=n_featuremaps,
                                                       ch_out=(n_featuremaps // 2),
                                                       normalization=self.normalization))
            self.decoder2Conv.append(ConvBlock(ch_in=n_featuremaps,
                                               ch_out=(n_featuremaps // 2),
                                               act_fun=self.act_fun,
                                               normalization=self.normalization))
            n_featuremaps //= 2

        # Last 1x1 convolutions
        self.decoder1Conv.append(nn.Conv2d(n_featuremaps, 3, kernel_size=1, stride=1, padding=0))
        self.decoder2Conv.append(nn.Conv2d(n_featuremaps, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()

        # Encoder
        for i in range(len(self.encoderConv) - 1):
            x = self.encoderConv[i](x)
            x_temp.append(x)
            if self.pool_method == 'max':
                x = self.pooling(x)
            elif self.pool_method == 'conv':
                x = self.pooling[i](x)
        x = self.encoderConv[-1](x)

        # Intermediate results for concatenation
        x_temp = list(reversed(x_temp))

        # Decoder 1 (borders + seeds)
        for i in range(len(self.decoder1Conv) - 1):
            if i == 0:
                x1 = self.decoder1Upconv[i](x)
            else:
                x1 = self.decoder1Upconv[i](x1)
            x1 = torch.cat([x1, x_temp[i]], 1)
            x1 = self.decoder1Conv[i](x1)
        x1 = self.decoder1Conv[-1](x1)

        # Decoder 2 (cells)
        for i in range(len(self.decoder2Conv) - 1):
            if i == 0:
                x2 = self.decoder2Upconv[i](x)
            else:
                x2 = self.decoder2Upconv[i](x2)
            x2 = torch.cat([x2, x_temp[i]], 1)
            x2 = self.decoder2Conv[i](x2)
        x2 = self.decoder2Conv[-1](x2)

        return x1, x2
