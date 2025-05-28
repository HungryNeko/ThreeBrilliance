import math
from colorsys import yiq_to_rgb

import numpy as np
import random
from collections import defaultdict

from character import player


class STGAgent:
    def __init__(self):
        # 动作定义: [0:左上, 1:上, 2:右上, 3:左, 4:不动, 5:右, 6:左下, 7:下, 8:右下]
        self.action_size = 9
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.bullet_count=[]
        self.care_size=7

        # Q表 (使用字典存储)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

        # 状态参数
        self.safe_threshold = 2  # 安全区域的子弹数量阈值
        self.zone_width = 50  # 区域宽度

        # 奖励参数
        self.reward_params = {
            'hit': 5.0,  # 击中敌人
            'hurt': -10.0,  # 被击中
            'safe_zone': 1.0,  # 安全区域奖励
            'danger_zone': -2.0,  # 危险区域惩罚
            'enemy_side': 0.5,  # 朝向敌人方向奖励
            'move': 1.000,  # 移动奖励
            'wall_close' : -1.0  # 靠近墙壁惩罚
        }

        # 跟踪状态
        self.last_state = None
        self.last_action = None
        self.last_position = [0, 0]

    def get_action(self, processed_state):
        """根据状态选择动作"""
        state_key = self._state_to_key(processed_state)

        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)  # 随机选择方向

        # 获取Q值
        q_values = self.q_table[state_key]

        # 获取安全区域（子弹少的区域）
        bullet_counts, enemy_side = processed_state
        safe_actions = [i for i, count in enumerate(bullet_counts) if count < self.safe_threshold]

        # 优先选择安全区域
        if safe_actions:
            safe_q = [q_values[i] for i in safe_actions]
            return safe_actions[np.argmax(safe_q)]

        # 没有安全区域时考虑敌人水平方向
        if enemy_side is not None:
            # 获取与敌人方向一致的动作（左/右相关）
            if enemy_side == 'left':
                preferred_actions = [0, 3, 6]  # 左上,左,左下
            else:
                preferred_actions = [2, 5, 8]  # 右上,右,右下

            preferred_q = [q_values[i] for i in preferred_actions]

            # 如果Q值差距不大(10%)，优先选择敌人方向
            max_q = np.max(q_values)
            if max_q - np.max(preferred_q) < max_q * 0.1:
                return preferred_actions[np.argmax(preferred_q)]

        # 默认选择最高Q值动作
        return np.argmax(q_values)

    def learn(self, reward, next_state, done):
        """Q-learning更新"""
        if self.last_state is None:
            return

        last_state_key = self._state_to_key(self.last_state)
        next_state_key = self._state_to_key(next_state)

        # Q-learning更新公式
        current_q = self.q_table[last_state_key][self.last_action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q * (1 - done) - current_q)

        # 更新Q表
        self.q_table[last_state_key][self.last_action] = new_q

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _process_game_state(self, enemy_list, player_pos, hit, hurt, wall_x, wall_y):
        """处理游戏状态：
        返回: (各方向子弹数[9], 敌人在左/右), reward
        """
        x, y = player_pos
        d = self.care_size # 区域半径

        # 定义9个方向区域中心点
        directions = [
            (x - d, y - d), (x, y - d), (x + d, y - d),  # 上排
            (x - d, y), (x, y), (x + d, y),  # 中排
            (x - d, y + d), (x, y + d), (x + d, y + d)  # 下排
        ]

        # 计算每个方向的子弹数量
        bullet_counts = [0] * 9
        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if bullet.show:
                    b_x, b_y = bullet.record[0], bullet.record[1]
                    b_speed= bullet.record[2]
                    for i, (dir_x, dir_y) in enumerate(directions):
                        distance_to_direction = math.sqrt((b_x - dir_x) ** 2 + (b_y - dir_y) ** 2)
                        time_to_hit= distance_to_direction / b_speed
                        bullet_counts[i] += 1/(time_to_hit+0.1)
        self.bullet_count=bullet_counts


        # 确定敌人在左还是右
        enemy_side = None
        if enemy_list:
            closest_enemy = min(enemy_list, key=lambda e: abs(e.position_x - x))
            if closest_enemy.position_x < x:
                enemy_side = 'left'
            elif closest_enemy.position_x > x:
                enemy_side = 'right'

        # 计算奖励
        reward = hit * self.reward_params['hit'] + hurt * self.reward_params['hurt']

        # 安全区域奖励
        safe_zones = sum(1 for count in bullet_counts if count < self.safe_threshold)
        reward += safe_zones * self.reward_params['safe_zone']

        # 危险区域惩罚
        danger_zones = sum(1 for count in bullet_counts if count >= self.safe_threshold)
        reward += danger_zones * self.reward_params['danger_zone']

        # 朝向敌人奖励
        if enemy_side is not None:
            reward += self.reward_params['enemy_side']

        # 移动奖励
        if self.last_position != [x, y]:
            reward -= self.reward_params['move']
        else:
            reward += self.reward_params['move']

        self.last_position = [x, y]
        # 靠近墙壁惩罚
        x_close_to_wall = min(x, wall_x - x)/ (wall_x + 0.001)
        y_close_to_wall = min(y, wall_y - y)/ (wall_y + 0.001)
        reward-=(abs(0.5 - x_close_to_wall)+abs(0.5-y_close_to_wall)) * self.reward_params['wall_close']
        bullet_counts.append(x/(wall_x+0.001))
        bullet_counts.append(y/(wall_y+0.001))# 添加靠近墙壁的惩罚
        state = (bullet_counts, enemy_side)
        return state, reward

    def _state_to_key(self, state):
        """将状态转换为可哈希的键"""
        bullet_counts, enemy_side = state
        side_code = 0 if enemy_side == 'left' else 1 if enemy_side == 'right' else 2
        return tuple(bullet_counts) + (side_code,)

    def save(self, filename):
        """保存Q表"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename):
        """加载Q表"""
        import pickle
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), q_table)