import torch
from torch import nn, einsum
import math
from einops import rearrange
import numpy as np


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GSA_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, heads=3, timedim=64):
        super().__init__()
        dim_head = in_channels // 8
        self.heads = heads

        self.to_v = nn.ModuleList()
        for i in range(self.heads):
            self.to_v.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)))

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        y = None
        for i in range(self.heads):
            v = self.to_v[i](x)
            dots = self.PA[i]
            out = einsum('i j, b c t j -> b c t i', dots, v)
            y = out + y if y is not None else out
        return y


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class transformer_block(nn.Module):
    def __init__(self, A, in_channels=64, out_channels=128, residual=False, heads=3, timedim=64):
        super(transformer_block, self).__init__()

        self.space_att = GSA_block(in_channels=in_channels, out_channels=out_channels, A=A, heads=heads,
                                   timedim=timedim)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.norm(self.space_att(x)) + self.residual(x)
        return self.relu(out)


class Temporal_SG_Layer(nn.Module):
    def __init__(self, in_channels, out_channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = nn.ReLU(inplace=True)

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, 1, bias=bias),
            nn.BatchNorm2d(out_channel),
        )

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        return x + res


class CW_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True, ):
        super().__init__()

        branch_channels = out_channels // 4
        # Temporal Convolution branches
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            Temporal_SG_Layer(in_channels=in_channels // 2, out_channel=branch_channels, temporal_window_size=5,
                              bias=True, stride=stride, residual=False),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            Temporal_SG_Layer(in_channels=in_channels // 2, out_channel=branch_channels, temporal_window_size=9,
                              bias=True, stride=stride, residual=False),
        ))
        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, timedim=64):
        super(basic_block, self).__init__()

        self.transformer = transformer_block(A=A, in_channels=in_channels, out_channels=out_channels, heads=3,
                                             residual=residual, timedim=timedim)

        self.tcn1 = CW_TCN(out_channels, out_channels, stride=stride, residual=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        b, c, t, _ = x.shape
        out = self.transformer(x)
        out = self.tcn1(out) + self.residual(x)
        out = self.relu(out)
        return out


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 dropout=0., Is_joint=True, K=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.Is_joint = Is_joint
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l2 = basic_block(base_channel, base_channel, A, timedim=64)
        self.l3 = basic_block(base_channel, base_channel, A, timedim=64)
        self.l4 = basic_block(base_channel, base_channel, A, timedim=64)
        self.l5 = basic_block(base_channel, base_channel * 2, A, stride=2, timedim=64)
        self.l6 = basic_block(base_channel * 2, base_channel * 2, A, timedim=32)
        self.l7 = basic_block(base_channel * 2, base_channel * 2, A, timedim=32)
        self.l8 = basic_block(base_channel * 2, base_channel * 4, A, stride=2, timedim=32)
        self.l9 = basic_block(base_channel * 4, base_channel * 4, A, timedim=16)
        self.l10 = basic_block(base_channel * 4, base_channel * 4, A, timedim=16)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if dropout:
            self.drop_out = nn.Dropout(dropout)
        else:
            self.drop_out = lambda x: x

        self.K = K
        if K != 0:
            I = np.eye(25)
            self.A = torch.from_numpy(I - np.linalg.matrix_power(self.graph.A_outward_binary, K)).type(torch.float32)

        if self.Is_joint:
            self.input_map = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 2, 1),
                nn.BatchNorm2d(base_channel // 2),
                nn.LeakyReLU(0.1),
            )
            self.diff_map1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map2 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map3 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map4 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
        else:
            self.l1 = unit_tcn(in_channels, base_channel, kernel_size=1, stride=1)

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        if self.K != 0:
            x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
            x = self.A.to(x.device).expand(N * M * T, -1, -1) @ x
            x = rearrange(x, '(n m t) v c -> n c t v m', m=M, t=T)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if self.Is_joint:
            dif1 = x[:, :, 1:] - x[:, :, 0:-1]
            dif1 = torch.cat([dif1.new(N * M, C, 1, V).zero_(), dif1], dim=-2)
            dif2 = x[:, :, 2:] - x[:, :, 0:-2]
            dif2 = torch.cat([dif2.new(N * M, C, 2, V).zero_(), dif2], dim=-2)
            dif3 = x[:, :, :-1] - x[:, :, 1:]
            dif3 = torch.cat([dif3, dif3.new(N * M, C, 1, V).zero_()], dim=-2)
            dif4 = x[:, :, :-2] - x[:, :, 2:]
            dif4 = torch.cat([dif4, dif4.new(N * M, C, 2, V).zero_()], dim=-2)
            x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3),
                           self.diff_map4(dif4)), dim=1)
        else:
            x = self.l1(x)

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(10, 3, 64, 25, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model(graph='graph.ntu_rgb_d.Graph')
    net = net.to(device)
    img = img.to(device)
    x = net(img)
    # print(x.shape)
    print(count_parameters(net))
