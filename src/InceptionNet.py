import torch
import torch.nn as nn
import shutil
import src.NetworkUtils as utils


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        #self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.35, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=0)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()

        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()

        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()

        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2, padding=0)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2, padding=0)
        )

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()

        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=True),
            BasicConv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=0),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=0)
        )

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1, padding=0)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1, padding=0)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1, padding=0)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception(nn.Module):
    def __init__(self, num_classes=1001):
        super(Inception, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=0),                # features.0
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0),               # features.1
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),    # features.2
            Mixed_3a(),                                                 # features.3 .conv .conv/.bn
            Mixed_4a(),                                                 # 4
            Mixed_5a(),                                                 # 5
            Inception_A(),                                              # 6
            Inception_A(),
            Inception_A(),
            Inception_A(),                                              # 9
            Reduction_A(),
            Inception_B(),      #11
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),  # 17
            Reduction_B(),
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0, count_include_pad=False)
        self.Dropout = nn.Dropout(p=0.2)            # TODO: Dropout'u kaldır

    def logits(self, features):
        x = self.avgpool(features)
        x = self.Dropout(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input_img):
        x = self.features(input_img)
        x = self.logits(x)
        return x


def build_Inception():
    model = Inception()
    state_dict = torch.load('InceptionV4.pth')
    del state_dict["last_linear.bias"]
    del state_dict["last_linear.weight"]
    model.load_state_dict(state_dict, strict=False)
    for child in model.features.children():
        for param in child.parameters():
            param.requires_grad = False
    return model


def run(img_path):
    image_loader = utils.LoadImage()
    image_transformer = utils.TransformImage()
    original_img = image_loader(img_path)
    img_tensor = image_transformer(original_img)
    input_img = torch.autograd.Variable(img_tensor, requires_grad=False)
    input_img = input_img.unsqueeze(0)

    model = build_Inception()
    model.eval()
    out = model(input_img)
    print("Label Index: " + str(out.argmax()))
    print("Prob: " + str(out.max()))
    print(out.size())
    return out


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
