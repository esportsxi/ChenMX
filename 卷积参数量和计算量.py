import torch.nn as nn
import torch
from torchsummary import summary


class Conv_test(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, groups):
        super(Conv_test, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            groups=groups,
            bias=False
        )

    def forward(self, input):
        out = self.conv(input)
        return out

# 标准卷积
# 标准卷积，(3, 64, 64) -> (4, 64, 64)
# 参数量: in_ch * (k*k) * out_ch，则3x(3x3)x4 = 108
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = Conv_test(3, 4, 3, 1, 1).to(device)
print('————————————————————标准卷积——————————————————')
print(summary(conv,  input_size=(3, 64, 64)))

# 分组卷积
# 分组卷积层，(4, 64, 64) -> (6, 64, 64)
# 参数量: groups * (in_ch//groups) * (k*k) * (out_ch//groups)，则2x2x(3x3)x3 = 108
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = Conv_test(4, 6, 3, padding=1, groups=2).to(device)
print('————————————————————分组卷积——————————————————')
print(summary(conv,  input_size=(4, 64, 64)))

# 逐深度卷积
# 逐深度卷积，(3, 64, 64) -> (3, 64, 64)
# 参数量：1x(3x3)x3=27
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = Conv_test(3, 3, 3, padding=1, groups=3).to(device)
print('————————————————————逐深度卷积——————————————————')
print(summary(conv,  input_size=(3, 64, 64)))

# 逐点卷积，输入即逐深度卷积的输出大小，目标输出也是4个feature map (3, 64, 64) -> (4, 64, 64)
# 参数量: 3x(1x1)x4=12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = Conv_test(3, 4, kernel_size=1, padding=0, groups=1).to(device)
print('————————————————————逐点卷积——————————————————')
print(summary(conv,  input_size=(3, 64, 64)))

# 深度可分离卷积
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.dwconv(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建模型实例
model = myModel().to(device)
# 打印模型结构和参数量信息
print('————————————————————深度可分离卷积——————————————————')
summary(model, (3, 64, 64))
