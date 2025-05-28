import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
import os


class STGAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.target_update_freq = 1000
        self.learn_step_counter = 0
        self.save_freq = 1000
        self.last_position=[0,0]

        # 网络参数
        self.care_size = 20  # 每个区域跟踪的子弹数
        self.bullet_feats = 3  # [命中时间, 移动类型, 威力]

        # 初始化网络
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # 子弹类型跟踪
        self.b_types = {}

        # 奖励参数
        self.reward_params = {
            'hit': 1.0,
            'hurt': -1,
            'wall_penalty': -0.2,
            'enemy_penalty': 0.1,
            'survival': 0.000,
            'move':10
        }

        # 跟踪上一次的状态和动作
        self.last_state = None
        self.last_action = None

    def _build_network(self):
        """构建Dueling DQN网络"""

        class DuelingDQN(nn.Module):
            def __init__(self, care_size, bullet_feats, action_size):
                super(DuelingDQN, self).__init__()
                # 子弹编码器
                self.bullet_encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(9 * care_size * bullet_feats, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )

                # 敌人编码器
                self.enemy_encoder = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU()
                )

                # 位置编码器
                self.position_encoder = nn.Sequential(
                    nn.Linear(4, 32),
                    nn.ReLU()
                )

                # 合并层
                self.fc = nn.Sequential(
                    nn.Linear(128 + 32 + 32, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )

                # 价值流
                self.value_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

                # 优势流
                self.advantage_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_size)
                )

            def forward(self, bullet_input, enemy_input, position_input):
                # 处理各输入
                bullet_feats = self.bullet_encoder(bullet_input)
                enemy_feats = torch.mean(self.enemy_encoder(enemy_input), dim=1)
                position_feats = self.position_encoder(position_input)

                # 合并特征
                combined = torch.cat([bullet_feats, enemy_feats, position_feats], dim=1)
                hidden = self.fc(combined)

                # 计算价值和优势
                value = self.value_stream(hidden)
                advantage = self.advantage_stream(hidden)

                # 合并结果
                qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
                return qvals

        return DuelingDQN(self.care_size, self.bullet_feats, self.action_size)

    def get_action(self, processed_state):
        """
        根据处理后的状态获取动作
        参数:
            processed_state: 已处理的状态 (enemy_positions, wall_x, wall_y, x, y, zone_bullets)
        返回:
            action: 选择的动作 (0-8)
        """
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # 转换为张量
        bullet_tensor, enemy_tensor, position_tensor = self._state_to_tensor(processed_state)

        with torch.no_grad():
            q_values = self.policy_net(bullet_tensor, enemy_tensor, position_tensor)
            return q_values.argmax().item()

    def _process_game_state(self, enemy_list, player_pos, hit, hurt, wall_x, wall_y):
        """
        处理原始游戏状态为RL格式并计算奖励
        参数:
            enemy_list: 敌人列表
            player_pos: 玩家位置 [x, y]
            hit: 是否命中敌人
            hurt: 是否被击中
            wall_x: 墙壁宽度
            wall_y: 墙壁高度
        返回:
            processed_state: 处理后的状态
            reward: 计算的奖励
        """
        x, y = player_pos
        d = 3
        d1 = d * math.sqrt(2) / 2

        # 定义玩家周围的9个区域
        zones = [
            [x - d1, y - d1],  # 左上
            [x - d, y],  # 左
            [x - d1, y + d1],  # 左下
            [x, y - d],  # 上
            [x, y],  # 中
            [x, y + d],  # 下
            [x + d1, y - d1],  # 右上
            [x + d, y],  # 右
            [x + d1, y + d1]  # 右下
        ]

        zone_bullets = {i: [] for i in range(9)}
        enemy_positions = []

        # 处理敌人和子弹
        for enemy in enemy_list:
            if enemy.show:
                enemy_positions.append([enemy.position_x, enemy.position_y])

            for bullet in enemy.bullets:
                if bullet.move_func not in self.b_types:
                    self.b_types[bullet.move_func] = len(self.b_types) + 1

                if bullet.show:
                    b_x = bullet.record[0]
                    b_y = bullet.record[1]
                    b_speed = bullet.record[2]
                    distance = math.sqrt((b_x - x) ** 2 + (b_y - y) ** 2)
                    move_x = b_x - bullet.last_x
                    move_y = b_y - bullet.last_y

                    for zone, pos in enumerate(zones):
                        if self._is_bullet_dangerous(pos[0], pos[1], b_x, b_y, move_x, move_y):
                            hit_time = distance / (b_speed * 0.2 + 1e-6)
                            zone_bullets[zone].append([hit_time, self.b_types[bullet.move_func], bullet.power])

        # 标准化子弹信息
        for zone in zone_bullets:
            if len(zone_bullets[zone]) > self.care_size:
                zone_bullets[zone] = zone_bullets[zone][:self.care_size]
            else:
                zone_bullets[zone].extend([[0, 0, 0]] * (self.care_size - len(zone_bullets[zone])))

        # 计算奖励
        reward = self._calculate_reward(hit, hurt, x, y, wall_x, wall_y, enemy_positions)

        # 返回处理后的状态
        processed_state = (enemy_positions, wall_x, wall_y, x, y, zone_bullets)
        return processed_state, reward

    def learn(self, last_state, last_action, reward, current_state, done):
        """
        学习过程
        参数:
            last_state: 上一个状态
            last_action: 上一个动作
            reward: 获得的奖励
            current_state: 当前状态
            done: 是否结束
        """
        # 存储经验
        self.memory.append((last_state, last_action, reward, current_state, done))

        # 更新目标网络
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 定期保存模型
        if self.learn_step_counter % self.save_freq == 0:
            self.save('model_checkpoint.pth')

        # 经验回放
        if len(self.memory) >= self.batch_size:
            self._replay()

        self.learn_step_counter += 1

        # ε衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay(self):
        """经验回放"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 准备批量数据
        bullet_inputs = []
        enemy_inputs = []
        position_inputs = []
        next_bullet_inputs = []
        next_enemy_inputs = []
        next_position_inputs = []

        for state in states:
            b, e, p = self._state_to_tensor(state, False)
            bullet_inputs.append(b)
            enemy_inputs.append(e)
            position_inputs.append(p)

        for next_state in next_states:
            b, e, p = self._state_to_tensor(next_state, False)
            next_bullet_inputs.append(b)
            next_enemy_inputs.append(e)
            next_position_inputs.append(p)

        bullet_tensor = torch.cat(bullet_inputs, dim=0).to(self.device)
        enemy_tensor = torch.cat(enemy_inputs, dim=0).to(self.device)
        position_tensor = torch.cat(position_inputs, dim=0).to(self.device)

        next_bullet_tensor = torch.cat(next_bullet_inputs, dim=0).to(self.device)
        next_enemy_tensor = torch.cat(next_enemy_inputs, dim=0).to(self.device)
        next_position_tensor = torch.cat(next_position_inputs, dim=0).to(self.device)

        # 转换为张量
        action_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 当前Q值
        current_q = self.policy_net(bullet_tensor, enemy_tensor, position_tensor).gather(1, action_tensor)

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_bullet_tensor, next_enemy_tensor, next_position_tensor).max(1)[0].unsqueeze(1)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calculate_reward(self, hit, hurt, x, y, wall_x, wall_y, enemy_positions):
        """计算综合奖励"""

        # 基础奖励
        reward = hit * self.reward_params['hit'] + hurt * self.reward_params['hurt']

        # 墙壁惩罚
        wall_dist = min(x, wall_x - x, y, wall_y - y)
        reward -=abs(wall_dist * self.reward_params['wall_penalty'])

        # 敌人惩罚
        if enemy_positions:
            closest_dist = min(math.sqrt((e[0] - x) ** 2 + (e[1] - y) ** 2) for e in enemy_positions)
            reward -= abs(closest_dist * self.reward_params['enemy_penalty'])

        # 生存奖励
        reward += self.reward_params['survival']
        #print(x,y, self.last_position)
        if self.last_position!=[x,y]:
            reward += self.reward_params['move']
        else:
            reward -= self.reward_params['move']
        self.last_position= [x, y]
        return reward

    def _is_bullet_dangerous(self, p_x, p_y, b_x, b_y, move_x, move_y, hit_radius=5):
        """判断子弹是否对区域有威胁"""
        if move_x == 0 and move_y == 0:
            return False

        # 子弹方向
        move_mag = math.sqrt(move_x ** 2 + move_y ** 2)
        dir_x = move_x / move_mag
        dir_y = move_y / move_mag

        # 区域到子弹的向量
        zone_dir_x = p_x - b_x
        zone_dir_y = p_y - b_y

        # 距离子弹路径的距离
        distance = abs(dir_y * p_x - dir_x * p_y + dir_x * b_y - dir_y * b_x)

        # 子弹是否朝向区域
        is_facing = (dir_x * zone_dir_x + dir_y * zone_dir_y) > 0

        return (distance < hit_radius) and is_facing

    def _state_to_tensor(self, state, add_batch_dim=True):
        """将处理后的状态转换为张量"""
        enemy_positions, wall_x, wall_y, x, y, zone_bullets = state

        # 子弹信息
        bullet_array = np.zeros((9, self.care_size, 3))
        for zone in range(9):
            for i, bullet in enumerate(zone_bullets[zone]):
                bullet_array[zone, i] = bullet

        bullet_tensor = torch.FloatTensor(bullet_array)
        if add_batch_dim:
            bullet_tensor = bullet_tensor.unsqueeze(0)

        # 敌人位置
        enemy_array = np.array(enemy_positions) if enemy_positions else np.zeros((1, 2))
        enemy_tensor = torch.FloatTensor(enemy_array)
        if add_batch_dim:
            enemy_tensor = enemy_tensor.unsqueeze(0)

        # 位置信息
        position_array = np.array([x, y, wall_x, wall_y])
        position_tensor = torch.FloatTensor(position_array)
        if add_batch_dim:
            position_tensor = position_tensor.unsqueeze(0)

        return bullet_tensor.to(self.device), enemy_tensor.to(self.device), position_tensor.to(self.device)

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'b_types': self.b_types,
            'learn_step': self.learn_step_counter
        }, path)

    def load(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_state'])
            self.target_net.load_state_dict(checkpoint['target_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.b_types = checkpoint['b_types']
            self.learn_step_counter = checkpoint['learn_step']
            self.target_net.eval()