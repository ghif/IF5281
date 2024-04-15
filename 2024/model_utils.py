import torch
from torch import nn

# LeNet5 (original LeCun et al., 1998)
class LeNet5(nn.Module):
    def __init__(self, c, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        
        fc = nn.Linear(256, 120)
        relu = nn.ReLU()
        fc1 = nn.Linear(120, 84)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(84, num_classes)

        self.ff_block = nn.Sequential(*[fc, relu, fc1, relu1, fc2])
        
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = torch.flatten(h, 1)
        logits = self.ff_block(h)
        return logits

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, replicate=False):
        super(ConvBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.replicate = replicate
        if replicate:
            self.layer_r = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        if self.replicate:
            out = self.layer_r(out)
        out = self.layer2(out)
        return out
        
# Define model
class VGG16(nn.Module):
    def __init__(self, c, num_classes=10):
        super(VGG16, self).__init__()

        nchannels1 = [c, 64, 128]
        layers = []
        for i in range(len(nchannels1) - 1):
            in_channels = nchannels1[i]
            out_channels = nchannels1[i + 1]

            layers.append(ConvBlock(in_channels, out_channels))

        nchannels2 = [128, 256, 512, 512]
        for i in range(len(nchannels2) - 1):
            in_channels = nchannels2[i]
            out_channels = nchannels2[i + 1]

            layers.append(ConvBlock(in_channels, out_channels, replicate=True))
        self.conv_blocks = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1*1*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.conv_blocks(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
# class VGG16(nn.Module):
#     def __init__(self, c, num_classes=10):
#         super(VGG16, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(1*1*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        # expansion: int = 1,
        # downsample: nn.Module = None,
    ):
        super().__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        # self.expansion = expansion
        self.downsample_layer = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """

            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )

        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, 
            # out_channels*self.expansion, 
            out_channels,
            kernel_size=3, 
            padding=1,
            bias=False
        )
        # self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        
        # print(f"identity: {identity.shape}, out: {out.shape}")
        out += identity
        out = self.relu(out)
        return  out

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def make_residual_layer(
        in_channels: int, 
        out_channels: int, 
        stride: int, 
        num_layer: int
    ):

    layers = []

    # First layer: need downsampling if applicable
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))

    for _ in range(1, num_layer):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, 
            img_channel: int, 
            in_channels: int,
            num_classes: int, 
            nlayers: list[int]
        ):
        super().__init__()

        in_channels = 64
        conv1 = nn.Conv2d(
            img_channel,
            in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False 

        )
        bn1 = nn.BatchNorm2d(in_channels)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.init_block = nn.Sequential(*[conv1, bn1, relu, maxpool])

        res_layer1 = make_residual_layer(in_channels, 64, 1, nlayers[0])
        res_layer2 = make_residual_layer(64, 128, 2, nlayers[1])
        res_layer3 = make_residual_layer(128, 256, 2, nlayers[2])
        res_layer4 = make_residual_layer(256, 512, 2, nlayers[3])

        self.res_layers = nn.Sequential(*[res_layer1, res_layer2, res_layer3, res_layer4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.init_block(x)
        h = self.res_layers(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        logits = self.fc(h)

        return logits
    
# Define model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, h_n = self.rnn(x)
        y = self.linear(h)
        return y
    
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, (h_n, c_n) = self.lstm(x)
        y = self.linear(h)
        return y
    
    def get_states_across_time(self, x):
        h_c = None
        h_list, c_list = list(), list()
        with torch.no_grad():
            for t in range(x.size(1)):
                h_c = self.lstm(x[:, [t], :], h_c)[1]
                h_list.append(h_c[0])
                c_list.append(h_c[1])
            h = torch.cat(h_list)
            c = torch.cat(c_list)
        return h, c