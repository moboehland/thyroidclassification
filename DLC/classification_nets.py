from argparse import ArgumentError
import torch.nn as nn
import torchvision.models as models
from pytorchcv.model_provider import get_model as ptcv_get_model


def get_classification_net(net_type, pretrained, output_classes):
    if "resnet" in net_type:
        net = getattr(models, net_type)(pretrained=pretrained)  # pretrained not possible together with , num_classes=output_classes
        net.fc = nn.Linear(net.fc.in_features, output_classes)  # Replace fully connected layer with layer with desired number of output classes
    elif "vgg" in net_type:
        net = getattr(models, net_type)(pretrained=pretrained) # pretrained not possible together with , num_classes=output_classes
        net.classifier[6] = nn.Linear(net.classifier[6].in_features, output_classes)
    elif "InceptionV4" == net_type:
        net = ptcv_get_model(net_type, pretrained=True)
        net.output.fc = nn.Linear(net.output.fc.in_features, output_classes)
    elif "efficientnet" in net_type:
        net = ptcv_get_model(net_type, pretrained=True)   
        net.output.fc = nn.Linear(net.output.fc.in_features, output_classes)     
    elif "mobilenet_v2" == net_type:
        net = getattr(models, net_type)(pretrained=pretrained) # pretrained not possible together with , num_classes=output_classes
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, output_classes)
    else:
        raise ArgumentError(f"net: {net_type} unknown!")
    return net
