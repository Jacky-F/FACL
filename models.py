from xception import Xception
from xception_fam import Xception as Xception_fam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from einops import rearrange


class FAM_Head(nn.Module):
    def __init__(self, size):
        super(FAM_Head, self).__init__()
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)
        self.conv1 = nn.Conv2d(6, 3, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.size = size


    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]
        x_freq_avg = F.adaptive_avg_pool2d(x_freq, (self.size, self.size))
        x_freq_max = F.adaptive_max_pool2d(x_freq, (self.size, self.size))
        x_freq_cat = torch.cat((x_freq_avg, x_freq_max), dim=1)
        att = self.conv1(x_freq_cat)
        att = self.bn1(att)
        att = torch.sigmoid(att)
        y = self._DCT_all_T @ att @ self._DCT_all    # [N, 3, 299, 299]
        out = y + x
        out = torch.sigmoid(out)
        return out

class FACL(nn.Module):
    def __init__(self, num_classes=1, mode='FAM'):
        super(FACL, self).__init__()
        self.num_classes = num_classes
        self.mode = mode

        # init branches
        if mode == 'Original' or mode == 'Decoder' or mode =='CLloss':
            self.init_xcep()

        if mode == 'Two_Stream' or mode == 'FAM':
            self.init_xcep()
            self.init_xcep_fam()

        # classifier
        if mode == "Two_Stream" or mode == 'FAM':
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(2048+2048, num_classes)
            self.dp = nn.Dropout(p=0.2)

            if mode == "Two_Stream":
                self.decoder = Decoder2(4096)
                self.mlp = nn.Sequential(
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 2048)
                )
        elif mode == "Original":
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(2048, num_classes)
            self.dp = nn.Dropout(p=0.5)
        elif mode == "Decoder":
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(2048, num_classes)
            self.dp = nn.Dropout(p=0.5)
            self.decoder = Decoder2(2048)
        else:
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(2048, num_classes)
            self.dp = nn.Dropout(p=0.5)
            self.mlp = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 2048)
            )


    def init_xcep(self):
        self.xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, False)

    def init_xcep_fam(self):
        self.xcep_fam = Xception_fam(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.xcep_fam.load_state_dict(state_dict, False)


    def similarity(self, x):
        # x.shape: b 2048 10 10
        patch_size = 5
        y = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) (h w)', h=patch_size, w=patch_size) # b 8192 25
        y = F.normalize(y, p=2, dim=1)
        y_T = y.transpose(1, 2) # b 25 8192
        s = y_T @ y # b 25 25
        s = (s + 1) / 2 # range 0 ~ 1
        return s

    def forward(self, x):
        mask = None
        if self.mode == 'Original' or self.mode == 'CLloss':
            fea_high = self.xcep.features(x)
            fea = self.relu(fea_high)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'Decoder':
            fea_high = self.xcep.features(x)
            fea = self.relu(fea_high)
            mask = self.decoder(fea)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'Two_Stream':
            fea_rgb = self.xcep.features(x)
            fea_freq = self.xcep_fam.features(x)

            fea = torch.cat((fea_rgb, fea_freq), dim=1)
            fea = self.relu(fea)
            mask = self.decoder(fea)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'FAM':
            fea_rgb = self.xcep.features(x)
            fea_freq = self.xcep_fam.features(x)
            fea = torch.cat((fea_rgb, fea_freq), dim=1)
            fea = self.relu(fea)
            fea = self._norm_fea(fea)
            y = fea

        f = self.dp(fea)
        f = self.fc(f)

        if self.mode == 'Original' or self.mode == 'Decoder' or self.mode == 'FAM':
            return y, f, mask
        elif self.mode == 'Two_Stream' or self.mode == 'CLloss':
            return self.mlp(y), f, mask

    def _norm_fea(self, fea):
        f = fea
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def get_xcep_state_dict(pretrained_path='./outputs/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    state_dict = {k[12:]:v for k, v in state_dict.items()}
    return state_dict

class Decoder(nn.Module):
    def __init__(self, in_channel):
        super(Decoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=1, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(1, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, mode=1):
        super(UpSampleLayer, self).__init__()
        self.input = nn.Parameter(torch.randn(in_channels))
        self.mode = mode

    def forward(self, x):
        if self.mode == 1:
            return nn.functional.interpolate(x, scale_factor=2)
        else:
            return nn.functional.interpolate(x, size=(299, 299))


class Decoder2(nn.Module):
    def __init__(self, in_channel):
        super(Decoder2, self).__init__()
        self.decode = nn.Sequential(
            ConvLayer(in_channel, 1024, kernel_size=3, stride=1),
            nn.ReLU(),
            UpSampleLayer(1024),
            ConvLayer(1024, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            UpSampleLayer(512),
            ConvLayer(512, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            ConvLayer(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            UpSampleLayer(256),
            ConvLayer(256, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            UpSampleLayer(128),
            ConvLayer(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            ConvLayer(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            UpSampleLayer(64, 0),
            ConvLayer(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            ConvLayer(64, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decode(x)
