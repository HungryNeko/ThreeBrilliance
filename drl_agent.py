import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ACTION_LABELS = (
    "left-up", "up", "right-up",
    "right", "right-down", "down",
    "left-down", "left", "stay",
)


@dataclass
class GridState:
    grid: np.ndarray
    vector: np.ndarray
    features: tuple

    def __len__(self):
        return len(self.features)

    def __getitem__(self, key):
        return self.features[key]


class ReplayBuffer:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def __len__(self):
        return len(self.items)

    def push(self, state, action, reward, next_state, done=False):
        self.items.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.items, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


class ConvDQN(nn.Module):
    def __init__(self, grid_channels, vector_size, action_size, grid_shape=(90, 60)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_shape[0], grid_shape[1])
            conv_size = int(np.prod(self.conv(dummy).shape[1:]))
        self.vector_net = nn.Sequential(
            nn.Linear(vector_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, grid, vector):
        conv_out = self.conv(grid).flatten(1)
        vector_out = self.vector_net(vector)
        return self.head(torch.cat([conv_out, vector_out], dim=1))


class DRLAgent:
    def __init__(self, device="auto"):
        self.action_size = 9
        self.wall_x = 600
        self.wall_y = 900
        self.direction_range = 10
        self.warning_range = 30
        self.threat_move_step = 25
        self.max_lookahead = 500
        self.wall_repulse_dist = 100
        self.threat_angle_degrees = 10
        self.shot_range = 900
        self.shot_cone_angle_degrees = 14
        self.shot_center_angle_degrees = 4
        self.use_hard_rules = False

        self.grid_width = 60
        self.grid_height = 90
        self.grid_channels = 8
        self.vector_size = 14
        self.device = self._select_device(device)

        self.gamma = 0.99
        self.epsilon = 0.20
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.train_every = 4
        self.target_update_every = 1000
        self.replay = ReplayBuffer(100_000)
        self.steps = 0
        self.updates = 0
        self.loss_ema = None

        self.policy_net = ConvDQN(
            self.grid_channels,
            self.vector_size,
            self.action_size,
            grid_shape=(self.grid_height, self.grid_width),
        ).to(self.device)
        self.target_net = ConvDQN(
            self.grid_channels,
            self.vector_size,
            self.action_size,
            grid_shape=(self.grid_height, self.grid_width),
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)

        self.last_state = None
        self.last_action = None
        self.last_decision = {}
        self.last_reward_components = {}

    @staticmethod
    def _select_device(device):
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_action(self, full_state, enemy_list=None):
        self.steps += 1
        q_values = self._predict_q(full_state)
        explored = False
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            explored = True
        else:
            best_value = float(np.max(q_values))
            best_actions = [i for i, value in enumerate(q_values) if abs(float(value) - best_value) < 1e-7]
            action = random.choice(best_actions)

        self.last_decision = {
            "policy_mode": "dqn-cnn",
            "hard_rules_enabled": False,
            "candidate_actions": tuple(range(self.action_size)),
            "action": action,
            "action_label": ACTION_LABELS[action],
            "epsilon": self.epsilon,
            "explored": explored,
            "q_values": tuple(float(v) for v in q_values),
            "action_priors": tuple(0.0 for _ in range(self.action_size)),
            "adjusted_values": tuple(float(v) for v in q_values),
            "device": str(self.device),
            "replay_size": len(self.replay),
            "updates": self.updates,
            "loss": self.loss_ema,
        }
        self.last_state = full_state
        self.last_action = action
        return action

    def learn(self, reward, next_full_state, action, done=False):
        if self.last_state is not None and self.last_action is not None:
            self.replay.push(self.last_state, self.last_action, reward, next_full_state, done)
        self.last_state = next_full_state
        self.last_action = action

        if len(self.replay) < self.batch_size or self.steps % self.train_every != 0:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        grid = self._batch_grids(states)
        vector = self._batch_vectors(states)
        next_grid = self._batch_grids(next_states)
        next_vector = self._batch_vectors(next_states)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy_net(grid, vector).gather(1, action_tensor)
        with torch.no_grad():
            next_actions = self.policy_net(next_grid, next_vector).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_grid, next_vector).gather(1, next_actions)
            target = reward_tensor + (1.0 - done_tensor) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        loss_value = float(loss.detach().cpu())
        self.loss_ema = loss_value if self.loss_ema is None else self.loss_ema * 0.98 + loss_value * 0.02
        self.updates += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.updates % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _predict_q(self, state):
        with torch.no_grad():
            grid = self._batch_grids([state])
            vector = self._batch_vectors([state])
            return self.policy_net(grid, vector).squeeze(0).detach().cpu().numpy()

    def _batch_grids(self, states):
        arr = np.stack([s.grid for s in states]).astype(np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _batch_vectors(self, states):
        arr = np.stack([s.vector for s in states]).astype(np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _process_game_state(self, enemy_list, player_pos, hit, hurt, wall_x, wall_y):
        self.wall_x = wall_x
        self.wall_y = wall_y
        x, y = player_pos

        bullet_counts = self._get_local_bullet_counts(enemy_list, x, y)
        threat_scores = self._get_direction_threat(x, y, wall_x, wall_y, enemy_list)
        norm_ex, norm_ey, enemy_count = self._closest_enemy_features(enemy_list, x, y, wall_x, wall_y)
        aim_features = self._get_shot_cone_features(x, y, wall_x, wall_y, enemy_list)
        grid = self._encode_grid(enemy_list, x, y, wall_x, wall_y)

        enemies_alive = any(getattr(e, "show", True) for e in enemy_list)
        wall_penalty_dist = 100
        wall_punish_base = 1.0
        k = 15.0
        min_dist_to_wall = min(x, wall_x - x, y, wall_y - y)
        if min_dist_to_wall < wall_penalty_dist:
            wall_penalty = wall_punish_base * math.exp((wall_penalty_dist - min_dist_to_wall) / k)
        else:
            wall_penalty = 0.0
        if min_dist_to_wall > 200:
            wall_reward_coef = 1.0
        elif min_dist_to_wall > 100:
            wall_reward_coef = 0.6
        elif min_dist_to_wall > 50:
            wall_reward_coef = 0.4
        else:
            wall_reward_coef = 0.2

        aim_left, aim_center, aim_right, aim_alignment, aim_dist = aim_features
        base_reward = hit * 100.0 + hurt * -100.0
        aim_reward = (aim_center * 2.0 + (aim_left + aim_right) * 0.6 + aim_alignment * 2.0) * 0.5
        wall_reward = -wall_penalty * (1.0 if enemies_alive else 3.0)
        reward = (base_reward + aim_reward + wall_reward) * wall_reward_coef
        self.last_reward_components = {
            "hit": hit * 100.0,
            "hurt": hurt * -100.0,
            "aim": aim_reward,
            "wall": wall_reward,
            "wall_coef": wall_reward_coef,
            "total": reward,
        }

        bullet_total = sum(
            1
            for enemy in enemy_list
            for bullet in enemy.bullets
            if getattr(bullet, "show", False)
        )
        vector = np.array(
            [
                x / wall_x,
                y / wall_y,
                norm_ex,
                norm_ey,
                min(1.0, enemy_count / 10.0),
                min(1.0, bullet_total / 300.0),
                min(1.0, max(0.0, hit / 100.0)),
                min(1.0, max(0.0, hurt / 100.0)),
                min(1.0, aim_left / 3.0),
                min(1.0, aim_center / 3.0),
                min(1.0, aim_right / 3.0),
                aim_alignment,
                aim_dist,
                min(1.0, min_dist_to_wall / 200.0),
            ],
            dtype=np.float32,
        )
        features = (
            *bullet_counts,      # 0-8
            *threat_scores,      # 9-17
            x, y,                # 18-19
            norm_ex, norm_ey,    # 20-21
            wall_x, wall_y,      # 22-23
            *aim_features,       # 24-28
        )
        return GridState(grid=grid, vector=vector, features=features), reward

    def _encode_grid(self, enemy_list, x, y, wall_x, wall_y):
        grid = np.zeros((self.grid_channels, self.grid_height, self.grid_width), dtype=np.float32)
        sx = self.grid_width / wall_x
        sy = self.grid_height / wall_y

        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, "show", False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                if not (0 <= bx <= wall_x and 0 <= by <= wall_y):
                    continue
                radius = max(1, int(math.ceil(getattr(bullet, "size", 2) * max(sx, sy))))
                gx, gy = int(bx * sx), int(by * sy)
                self._stamp(grid[0], gx, gy, radius, 1.0)
                vx = getattr(bullet, "vx", bullet.position_x - getattr(bullet, "last_x", bullet.position_x))
                vy = getattr(bullet, "vy", bullet.position_y - getattr(bullet, "last_y", bullet.position_y))
                self._stamp(grid[1], gx, gy, radius, max(-1.0, min(1.0, vx / 20.0)))
                self._stamp(grid[2], gx, gy, radius, max(-1.0, min(1.0, vy / 20.0)))

        for enemy in enemy_list:
            if not getattr(enemy, "show", True):
                continue
            ex, ey = enemy.position_x, enemy.position_y
            gx, gy = int(ex * sx), int(ey * sy)
            radius = max(1, int(math.ceil(getattr(enemy, "size", 5) * max(sx, sy))))
            hp_norm = min(1.0, max(0.05, getattr(enemy, "health", 1) / max(getattr(enemy, "full_health", 1), 1)))
            self._stamp(grid[3], gx, gy, radius, hp_norm)
            if getattr(enemy, "boss", False):
                self._stamp(grid[4], gx, gy, radius + 1, hp_norm)

        px, py = int(x * sx), int(y * sy)
        self._stamp(grid[5], px, py, 1, 1.0)
        self._encode_shot_cone_channel(grid[6], x, y, wall_x, wall_y)
        self._encode_wall_channel(grid[7])
        return grid

    @staticmethod
    def _stamp(channel, gx, gy, radius, value):
        height, width = channel.shape
        x0, x1 = max(0, gx - radius), min(width - 1, gx + radius)
        y0, y1 = max(0, gy - radius), min(height - 1, gy + radius)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - gx) ** 2 + (yy - gy) ** 2 <= radius ** 2:
                    if value >= 0:
                        channel[yy, xx] = max(channel[yy, xx], value)
                    else:
                        channel[yy, xx] = min(channel[yy, xx], value)

    def _encode_shot_cone_channel(self, channel, x, y, wall_x, wall_y):
        cone = math.radians(self.shot_cone_angle_degrees)
        xs = (np.arange(self.grid_width, dtype=np.float32) + 0.5) * wall_x / self.grid_width
        ys = (np.arange(self.grid_height, dtype=np.float32) + 0.5) * wall_y / self.grid_height
        xx, yy = np.meshgrid(xs, ys)
        dx = xx - x
        dy_up = y - yy
        dist = np.sqrt(dx * dx + dy_up * dy_up)
        angle = np.abs(np.arctan2(dx, np.maximum(dy_up, 1e-3)))
        mask = (dy_up > 0) & (dist <= self.shot_range) & (angle <= cone)
        channel[mask] = 1.0 - np.minimum(1.0, angle[mask] / cone) * 0.5

    def _encode_wall_channel(self, channel):
        yy, xx = np.indices(channel.shape)
        left = xx
        right = self.grid_width - 1 - xx
        top = yy
        bottom = self.grid_height - 1 - yy
        dist = np.minimum(np.minimum(left, right), np.minimum(top, bottom)).astype(np.float32)
        channel[:] = 1.0 - np.minimum(1.0, dist / 10.0)

    def _get_local_bullet_counts(self, enemy_list, x, y):
        d = self.direction_range
        diag = d / math.sqrt(2)
        directions = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y),
        ]
        counts = [0] * 9
        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, "show", False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                bsize = getattr(bullet, "size", 0)
                for i, (zone_x, zone_y) in enumerate(directions):
                    if math.hypot(bx - zone_x, by - zone_y) <= d + bsize:
                        counts[i] += 1
        return counts

    def _closest_enemy_features(self, enemy_list, x, y, wall_x, wall_y):
        enemies = [e for e in enemy_list if getattr(e, "show", True)]
        if not enemies:
            return 0.5, 0.5, 0
        closest = min(enemies, key=lambda e: abs(e.position_x - x) + abs(e.position_y - y))
        rel_ex = max(-wall_x, min(wall_x, closest.position_x - x))
        rel_ey = max(-wall_y, min(wall_y, closest.position_y - y))
        return (rel_ex / wall_x + 1) / 2, (rel_ey / wall_y + 1) / 2, len(enemies)

    def _get_shot_cone_features(self, x, y, wall_x, wall_y, enemy_list):
        left_count = 0
        center_count = 0
        right_count = 0
        best_alignment = 0.0
        nearest_dist_norm = 1.0
        cone_angle = math.radians(self.shot_cone_angle_degrees)
        center_angle = math.radians(self.shot_center_angle_degrees)

        for enemy in enemy_list:
            if not getattr(enemy, "show", True):
                continue
            dx = enemy.position_x - x
            dy_up = y - enemy.position_y
            if dy_up <= 0:
                continue
            dist = math.hypot(dx, dy_up)
            if dist > self.shot_range:
                continue
            angle = math.atan2(dx, dy_up)
            enemy_radius_angle = math.atan2(getattr(enemy, "size", 0), max(dy_up, 1.0))
            effective_abs_angle = max(0.0, abs(angle) - enemy_radius_angle)
            if effective_abs_angle > cone_angle:
                continue
            if angle < -center_angle:
                left_count += 1
            elif angle > center_angle:
                right_count += 1
            else:
                center_count += 1
            alignment = 1.0 - min(1.0, effective_abs_angle / cone_angle)
            best_alignment = max(best_alignment, alignment)
            nearest_dist_norm = min(nearest_dist_norm, min(1.0, dist / self.shot_range))
        return left_count, center_count, right_count, best_alignment, nearest_dist_norm

    def _get_direction_threat(self, x, y, wall_x, wall_y, enemy_list):
        action_delta = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        ]
        threat_scores = [0.0] * 9
        for dir_idx, (dx, dy) in enumerate(action_delta):
            if dx == 0 and dy == 0:
                threat_scores[dir_idx] = 1e9
                continue
            dir_angle = math.atan2(dy, dx)
            score = 0.0
            for enemy in enemy_list:
                for bullet in enemy.bullets:
                    if not getattr(bullet, "show", False):
                        continue
                    bx, by = bullet.position_x, bullet.position_y
                    bvx = getattr(bullet, "vx", bullet.position_x - getattr(bullet, "last_x", bullet.position_x))
                    bvy = getattr(bullet, "vy", bullet.position_y - getattr(bullet, "last_y", bullet.position_y))
                    speed = math.hypot(bvx, bvy)
                    if speed < 1e-3:
                        continue
                    bullet_angle = math.atan2(bvy, bvx)
                    angle_diff = abs(self._angle_diff(bullet_angle, dir_angle))
                    if angle_diff > math.radians(self.threat_angle_degrees):
                        continue
                    dist = math.hypot(bx - x, by - y)
                    if dist > self.max_lookahead:
                        continue
                    score += (self.max_lookahead - dist) / speed
            new_x = x + dx * self.threat_move_step
            new_y = y + dy * self.threat_move_step
            min_dist_to_wall = min(new_x, wall_x - new_x, new_y, wall_y - new_y)
            if min_dist_to_wall < self.wall_repulse_dist:
                score += 20.0 * math.exp((self.wall_repulse_dist - min_dist_to_wall) / 15.0)
            threat_scores[dir_idx] = score
        return threat_scores

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def save(self, filename):
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "updates": self.updates,
                "loss_ema": self.loss_ema,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.steps = checkpoint.get("steps", self.steps)
        self.updates = checkpoint.get("updates", self.updates)
        self.loss_ema = checkpoint.get("loss_ema", self.loss_ema)
