import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_loss():
    """

    :param loss_function:
    :param unet:
    :return:
    """

    criterion_border = ce_dice
    criterion_cell = bce_dice

    criterion = {'border': criterion_border, 'cell': criterion_cell}

    return criterion


def dice_loss(y_pred, y_true, use_sigmoid=True):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :param use_sigmoid: Apply sigmoid activation function to the prediction y_pred.
        :type use_sigmoid: bool
    :return:
    """

    # Avoid division by zero
    smooth = 1.

    # Flatten ground truth
    gt = y_true.contiguous().view(-1)

    if use_sigmoid:  # Apply sigmoid activation to prediction and flatten prediction
        pred = torch.sigmoid(y_pred)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred.contiguous().view(-1)

    # Calculate Dice loss
    pred_gt = torch.sum(gt * pred)
    loss = 1 - (2. * pred_gt + smooth) / (torch.sum(gt ** 2) + torch.sum(pred ** 2) + smooth)

    return loss


def bce_dice(y_pred, y_true):
    """ Sum of binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :return:
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)
    return loss


def wbce_dice(y_pred, y_true):
    """ Sum of weighted binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :return:
    """

    eps = 1e-9

    w0 = 1 / torch.sqrt(torch.sum(y_true) + eps) * y_true
    w1 = 1 / torch.sqrt(torch.sum(1 - y_true)) * (1 - y_true)

    weight_map = w0 + w1
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = gaussian_smoothing_2d(weight_map, 1, 5, 2)

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true))
    loss = bce_loss + 2 * dice_loss(y_pred, y_true)

    return loss


def ce_dice(y_pred, y_true, num_classes=3):
    """Sum of crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width]. (channels=1)
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = nn.functional.one_hot(y_true[:,0,:,:], num_classes).float()
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    # Crossentropy Loss
    loss_ce = nn.CrossEntropyLoss()
    ce_loss = loss_ce(y_pred, y_true[:,0,:,:])

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :], y_true_one_hot[:, index, :, :],
                                        use_sigmoid=False)

    return ce_loss + 0.5 * dice_score


def wce_dice(y_pred, y_true, num_classes=3):
    """Sum of weighted crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = nn.functional.one_hot(y_true, num_classes).float()
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    eps = 1e-9

    # Weighted CrossEntropy Loss
    w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :]) + eps) * y_true_one_hot[:, 0, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :]) + eps) * y_true_one_hot[:, 1, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :]) + eps) * y_true_one_hot[:, 2, :, :]

    weight_map = w0 + w1 + w2
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :]

    weight_map = gaussian_smoothing_2d(weight_map, 1, 5, 0.9)

    loss_ce = nn.CrossEntropyLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_ce(y_pred, y_true))

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :], y_true_one_hot[:, index, :, :],
                                        use_sigmoid=False)

    return 0.5 * ce_loss + 0.4 * dice_score


def gaussian_smoothing_2d(x, channels, kernel_size, sigma):

    kernel_size = [kernel_size] * 2
    sigma = [sigma] * 2

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=2, bias=False)
    conv = conv.to('cuda')
    conv.weight.data = kernel.to('cuda')
    conv.weight.requires_grad = False

    return conv(x)
