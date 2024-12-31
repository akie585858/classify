from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

class ResidualLayer(nn.Module):
    def __init__(self, in_c:int, plain_c:int, deep=False):
        super().__init__()

        if not deep:
            self.net = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(True),
                nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_c),
            )
        else:
            out_c = 4*plain_c
            self.net = nn.Sequential(
                nn.Conv2d(in_c, plain_c, 1, bias=False),
                nn.BatchNorm2d(plain_c),
                nn.ReLU(True),
                nn.Conv2d(plain_c, plain_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(plain_c),
                nn.ReLU(True),
                nn.Conv2d(plain_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        x = self.net(x) + x
        return F.relu(x, True)


class DownLayer(nn.Module):
    def __init__(self, in_c:int, plain_c:int, deep=False):
        super().__init__()

        if deep:
            out_c = 4*plain_c
            self.net = nn.Sequential(
                nn.Conv2d(in_c, plain_c, 1, bias=False),
                nn.BatchNorm2d(plain_c),
                nn.ReLU(True),
                nn.Conv2d(plain_c, plain_c, 3, stride=2, padding=1 ,bias=False),
                nn.BatchNorm2d(plain_c),
                nn.ReLU(True),
                nn.Conv2d(plain_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            out_c = plain_c
            self.net = nn.Sequential(
                nn.Conv2d(in_c, plain_c, 3, stride=2, padding=1,bias=False),
                nn.BatchNorm2d(plain_c),
                nn.ReLU(True),
                nn.Conv2d(plain_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
            )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        y = self.net(x)
        short = self.shortcut(x)

        x = y + short
        return F.relu(x, True)


class ResNet(nn.Module):
    def __init__(self, init_c:int, cls_nums:int, high_map:bool=False, deep:bool=False, *layers):
        super().__init__()

        if high_map:
            self.head = nn.Sequential(
                nn.Conv2d(3, init_c, 7, 2, 3, bias=False),
                nn.BatchNorm2d(init_c),
                nn.ReLU(True),
                nn.MaxPool2d(3, 2, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(3, init_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(init_c),
                nn.ReLU(True)
            )
        
        # 初始化各种通道
        c = init_c
        plain_c = init_c
        out_c = init_c
        self.body = nn.Sequential()

    
        if deep:
            out_c *= 4
        
        for layer_nums in layers:
            if c != out_c:
                self.body.append(DownLayer(c, plain_c, deep))
            else:
                self.body.append(ResidualLayer(c, plain_c, deep))
            c = out_c
            for _ in range(layer_nums-1):
                self.body.append(ResidualLayer(c, plain_c, deep))
            plain_c *= 2
            out_c *= 2
        
        # 分类头
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_c//2, cls_nums)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.classify(x)

# img224, cls100
def resnet18_img224_cls100():
    return ResNet(64, 100, True, False, 2, 2, 2, 2)

def resnet34_img224_cls100():
    return ResNet(64, 100, True, False, 3, 4, 6, 3)

def resnet50_img224_cls100():
    return ResNet(64, 100, True, True, 3, 4, 6, 3)

def resnet101_img224_cls100():
    return ResNet(64, 100, True, True, 3, 4, 23, 3)

def resnet152_img224_cls100():
    return ResNet(64, 100, True, True, 3, 8, 36, 3)

# img224, cls10
def resnet18_img224_cls10():
    return ResNet(64, 10, True, False, 2, 2, 2, 2)

def resnet34_img224_cls10():
    return ResNet(64, 10, True, False, 3, 4, 6, 3)

def resnet50_img224_cls10():
    return ResNet(64, 10, True, True, 3, 4, 6, 3)

def resnet101_img224_cls10():
    return ResNet(64, 10, True, True, 3, 4, 23, 3)

def resnet152_img224_cls10():
    return ResNet(64, 10, True, True, 3, 8, 36, 3)

# img32, cls100, base
def resnet_img32_cls100_base(n:int, deep=False):
    return ResNet(16, 100, False, deep, 2*n, 2*n, 2*n)

# img32, cls100, bottleneck
def resnet_img32_cls100_bottleneck(n:int):
    return ResNet(16, 100, False, True, 2*n, 2*n, 2*n)

# img32, cls10, base
def resnet_img32_cls10_base(n:int):
    return ResNet(16, 10, False, False, 2*n, 2*n, 2*n)

# img32, cls10, bottleneck
def resnet_img32_cls10_bottleneck(n:int):
    return ResNet(16, 10, False, True, 2*n, 2*n, 2*n)


if __name__ == '__main__':
    import torch
    net = resnet_img32_cls10_bottleneck(3)

    for name, layer in net.named_modules():
        print(name)
    # x = torch.rand(5, 3, 32, 32)
    # y = net(x)
    # print(y.shape)