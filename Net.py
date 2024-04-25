
# Net.py：PPO 代理的神經網路。

import torch
import torch.nn as nn

# 共享網路
class CommonNetwork(nn.Module):
    def __init__(self):
        super(CommonNetwork, self).__init__()

        # 卷積塊1：9X9X4 -> 4X4X128
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 卷積塊2：4X4X128 -> 2X2X256
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 展平塊：2X2X256 -> 500X1
        self.flatten_block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=488, kernel_size=2, stride=1, padding=0),
            nn.Flatten(),
            nn.ReLU()
        )

        # 距離層
        self.distance_layer = nn.Linear(2, 12)
        # 位置層
        self.pos_layer = nn.Linear(2, 12)

        # 512X1 fully_connect 全連接塊：512X1
        self.fully_connected_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, obs, dis, pos): # 前向傳播
        x1 = self.conv_block1(obs)
        x2 = self.conv_block2(x1)
        x3 = self.flatten_block(x2)
        g1 = torch.relu(self.distance_layer(dis))
        p1 = torch.relu(self.distance_layer(pos))
        fc_in = torch.cat((x3, g1, p1), dim=1)
        fc_out = self.fully_connected_block(fc_in)
        return fc_out


# 策略網路
class PolicyNetwork(nn.Module):
    def __init__(self, common_network):
        super(PolicyNetwork, self).__init__()
        self.common_network = common_network
        # 全連接層
        self.fc = nn.Linear(512, 4)
        # 激活函數
        self.activation = nn.Softmax(dim=1)

    def forward(self, obs, dis, pos):
        # 前向傳播
        x = self.common_network(obs, dis, pos)
        fc1 = self.fc(x)
        return self.activation(fc1)

# 价值網路
class ValueNetwork(nn.Module):
    def __init__(self, common_network):
        super(ValueNetwork, self).__init__()
        self.common_network = common_network
        # 全連接層
        self.fc = nn.Linear(512, 1)

    def forward(self, obs, dis, pos): # 前向傳播
        x = self.common_network(obs, dis, pos)
        return self.fc(x)

# Actor-CriticNetwork
class ACNetwork(nn.Module):
    def __init__(self):
        super(ACNetwork, self).__init__()
        self.common_network0 = CommonNetwork()
        self.common_network1 = CommonNetwork()
        self.policy_network = PolicyNetwork(self.common_network0)
        self.value_network = ValueNetwork(self.common_network1)

    def forward(self, obs, dis, pos):
        return self.policy_network(obs, dis, pos), self.value_network(obs, dis)

    def forward_policy(self, obs, dis, pos):
        return self.policy_network(obs, dis, pos)

    def forward_value(self, obs, dis, pos):
        return self.value_network(obs, dis, pos)

