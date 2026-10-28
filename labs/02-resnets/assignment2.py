# @author: Aitana Cantó Rico
import torch

from PIL import Image
from torch import nn
from torchvision import transforms, models

def conv3x3(c_in:int, c_out:int, stride:int=1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(c_in:int, c_out:int, stride:int=1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in:int, c_out:int, stride:int=1) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # Task 1.1
        # First convolution 3x3
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm2d(c_out)

        # Second convolution 3x3
        self.conv2 = conv3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

        # Downsample if chanels or stride ≠ 1
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                conv1x1(c_in, c_out, stride),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Task 1.2
        input = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If there is downsample
        if hasattr(self, 'downsample'):
            input = self.downsample(x)

        out += input
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in:int, c_out:int, stride:int=1) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # Task 2.1
        chanels = c_out // self.expansion

        # 1x1 reduction
        self.conv1 = conv1x1(c_in, chanels)
        self.bn1 = nn.BatchNorm2d(chanels)

        # 3x3 processing
        self.conv2 = conv3x3(chanels, chanels, stride)
        self.bn2 = nn.BatchNorm2d(chanels)

        # 1x1 expansion
        self.conv3 = conv1x1(chanels, c_out)
        self.bn3 = nn.BatchNorm2d(c_out)

        # Downsample
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                conv1x1(c_in, c_out, stride),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Task 2.2
        input = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            input = self.downsample(x)

        out += input
        out = self.relu(out)
        return out
    
def make_resnet_layer(block:nn.Module, blocks:int, c_in:int, c_out:int, stride:int=1) -> nn.Module:
    # Task 3
    layers = []

    # 1rst block ->  stride ≠ 1 & c_in ≠ c_out
    layers.append(block(c_in, c_out, stride))

    # Other blocks -> c_out=c_in & stride = 1
    for _ in range(1, blocks):
        layers.append(block(c_out, c_out, stride=1))

    return nn.Sequential(*layers)



class ResNet(nn.Module):

    def __init__(self, block:nn.Module, layers:list[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Task 4.1
        self.layer1 = make_resnet_layer(block, layers[0], 64, 64 * block.expansion, stride=1)
        self.layer2 = make_resnet_layer(block, layers[1], 64 * block.expansion, 128 * block.expansion, stride=2)
        self.layer3 = make_resnet_layer(block, layers[2], 128 * block.expansion, 256 * block.expansion, stride=2)
        self.layer4 = make_resnet_layer(block, layers[3], 256 * block.expansion, 512 * block.expansion, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, 1000)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Task 4.2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":

    input_image = Image.open('dog.jpg')
    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    x = preprocess(input_image).unsqueeze(0)
    softmax = nn.Softmax(dim=1)
    resnet18_state_dict = torch.load('resnet18.pth')
    print('\nResNet18 parameters:')
    for k, v in resnet18_state_dict.items():
        print(k, v.shape)
    resnet50_state_dict = torch.load('resnet50.pth')
    print('\nResNet50 parameters:')
    for k, v in resnet50_state_dict.items():
        print(k, v.shape)

    # Test code

    resnet18 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    resnet18.eval()
    
    with torch.no_grad():
        prob = softmax(resnet18(x))
    top5_prob, top5_catid = torch.topk(prob, 5)
    print('\nPyTorch ResNet18 top-5 predictions:')
    for prob_i, catid_i in zip(top5_prob.squeeze(0), top5_catid.squeeze(0)):
        print(categories[catid_i.item()], prob_i.item())

    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet18.load_state_dict(resnet18_state_dict)
    resnet18.eval()
    with torch.no_grad():
        prob = softmax(resnet18(x))
    top5_prob, top5_catid = torch.topk(prob, 5)
    print('\nMy ResNet18 top-5 predictions:')
    for prob_i, catid_i in zip(top5_prob.squeeze(0), top5_catid.squeeze(0)):
        print(categories[catid_i.item()], prob_i.item())

    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
    resnet50.load_state_dict(resnet50_state_dict)
    resnet50.eval()
    with torch.no_grad():
        prob = softmax(resnet50(x))
    top5_prob, top5_catid = torch.topk(prob, 5)
    print('\nPyTorch ResNet50 top-5 predictions:')
    for prob_i, catid_i in zip(top5_prob.squeeze(0), top5_catid.squeeze(0)):
        print(categories[catid_i.item()], prob_i.item())

    resnet50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    resnet50.eval()
    with torch.no_grad():
        prob = softmax(resnet50(x))
    top5_prob, top5_catid = torch.topk(prob, 5)
    print('\nMy ResNet50 top-5 predictions:')
    for prob_i, catid_i in zip(top5_prob.squeeze(0), top5_catid.squeeze(0)):
        print(categories[catid_i.item()], prob_i.item())
  