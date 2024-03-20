import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # Initial layers
        #for 3 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #for 1 channel
        #self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512, 7)
        self.fc = self.fc.cuda()

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid() # not require for t.nn.BCEWithLogitsLoss()

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            ResBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        class_output = self.fc(x)
        class_output = self.sigmoid(class_output)

        # Perform segmentation task and get segmentation output
        seg_output = self.segmentation_layer(x)  # Example segmentation layer

        return class_output, seg_output


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class ResNetWithSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWithSegmentation, self).__init__()
        self.resnet = ResNet()
        
        # Segmentation head on top of layer3
        self.segmentation_head = SegmentationHead(256, num_classes)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        
        # Get features from layer3 for segmentation
        segmentation_features = self.resnet.layer3(x)
        segmentation_output = self.segmentation_head(segmentation_features)

        x = self.resnet.layer4(segmentation_features)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        x = self.resnet.sigmoid(x)

        return x, segmentation_output
