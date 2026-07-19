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
    local_grid: np.ndarray
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
    def __init__(
            self,
            grid_channels,
            local_grid_channels,
            vector_size,
            action_size,
            grid_shape=(90, 60),
            local_grid_shape=(64, 64)):
        super().__init__()
        self.global_conv = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_grid_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            global_dummy = torch.zeros(1, grid_channels, grid_shape[0], grid_shape[1])
            local_dummy = torch.zeros(1, local_grid_channels, local_grid_shape[0], local_grid_shape[1])
            global_conv_size = int(np.prod(self.global_conv(global_dummy).shape[1:]))
            local_conv_size = int(np.prod(self.local_conv(local_dummy).shape[1:]))
        self.vector_net = nn.Sequential(
            nn.Linear(vector_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(global_conv_size + local_conv_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, grid, local_grid, vector):
        global_out = self.global_conv(grid).flatten(1)
        local_out = self.local_conv(local_grid).flatten(1)
        vector_out = self.vector_net(vector)
        return self.head(torch.cat([global_out, local_out, vector_out], dim=1))


class DRLAgent:
    def __init__(self, device="auto", batch_size=64, train_every=4, gradient_steps=1):
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
        self.use_safety_prior = True
        self.player_radius = 5
        self.action_step = 10
        self.safety_margin = 22
        self.safety_lookahead_frames = (1, 2, 4, 6, 8)
        self.safety_penalty_scale = 6000.0
        self.use_attack_prior = True
        self.attack_prior_scale = 900.0
        self.boss_attack_prior_scale = 1800.0
        self.attack_vertical_min = 40.0
        self.attack_under_distance = 260.0
        self.attack_under_band = 180.0

        self.grid_width = 60
        self.grid_height = 90
        self.grid_channels = 8
        self.local_grid_size = 64
        self.local_grid_channels = 8
        self.local_world_radius = 260
        self.vector_size = 14
        self._grid_cache = {}
        self.device = self._select_device(device)

        self.gamma = 0.99
        self.epsilon = 0.20
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.9995
        self.batch_size = batch_size
        self.learning_rate = 1e-4
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.target_update_every = 1000
        self.replay = ReplayBuffer(100_000)
        self.steps = 0
        self.updates = 0
        self.loss_ema = None

        self.policy_net = ConvDQN(
            self.grid_channels,
            self.local_grid_channels,
            self.vector_size,
            self.action_size,
            grid_shape=(self.grid_height, self.grid_width),
            local_grid_shape=(self.local_grid_size, self.local_grid_size),
        ).to(self.device)
        self.target_net = ConvDQN(
            self.grid_channels,
            self.local_grid_channels,
            self.vector_size,
            self.action_size,
            grid_shape=(self.grid_height, self.grid_width),
            local_grid_shape=(self.local_grid_size, self.local_grid_size),
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
        enemy_list = enemy_list or ()
        safety_priors = self._action_safety_priors(full_state, enemy_list) if self.use_safety_prior else np.zeros(self.action_size, dtype=np.float32)
        attack_priors = self._action_attack_priors(full_state, enemy_list) if self.use_attack_prior else np.zeros(self.action_size, dtype=np.float32)
        action_priors = safety_priors + attack_priors
        adjusted_values = q_values + action_priors
        explored = False
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            explored = True
        else:
            best_value = float(np.max(adjusted_values))
            best_actions = [i for i, value in enumerate(adjusted_values) if abs(float(value) - best_value) < 1e-7]
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
            "action_priors": tuple(float(v) for v in action_priors),
            "safety_priors": tuple(float(v) for v in safety_priors),
            "attack_priors": tuple(float(v) for v in attack_priors),
            "adjusted_values": tuple(float(v) for v in adjusted_values),
            "device": str(self.device),
            "replay_size": len(self.replay),
            "updates": self.updates,
            "loss": self.loss_ema,
            "batch_size": self.batch_size,
            "train_every": self.train_every,
            "gradient_steps": self.gradient_steps,
        }
        self.last_state = full_state
        self.last_action = action
        return action

    def _action_attack_priors(self, state, enemy_list):
        if state is None or len(state) < 20:
            return np.zeros(self.action_size, dtype=np.float32)

        targets = [enemy for enemy in enemy_list if getattr(enemy, "show", True)]
        if not targets:
            return np.zeros(self.action_size, dtype=np.float32)

        bosses = [enemy for enemy in targets if getattr(enemy, "boss", False)]
        x = float(state[18])
        y = float(state[19])
        target = min(bosses or targets, key=lambda enemy: abs(enemy.position_x - x) + abs(enemy.position_y - y))
        dy_up = y - float(target.position_y)
        if dy_up > self.shot_range:
            return np.zeros(self.action_size, dtype=np.float32)

        scale = self.boss_attack_prior_scale if getattr(target, "boss", False) else self.attack_prior_scale
        cone_width = max(12.0, max(self.attack_vertical_min, dy_up) * math.tan(math.radians(self.shot_cone_angle_degrees)))
        current_error = abs(float(target.position_x) - x)
        current_under_error = abs(dy_up - self.attack_under_distance)
        action_delta = (
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        )
        priors = np.zeros(self.action_size, dtype=np.float32)
        for action, (adx, ady) in enumerate(action_delta):
            next_x = min(self.wall_x, max(0.0, x + adx * self.action_step))
            next_y = min(self.wall_y, max(0.0, y + ady * self.action_step))
            next_dy_up = next_y - float(target.position_y)
            if next_dy_up < self.attack_vertical_min:
                priors[action] -= scale * 0.5

            next_error = abs(float(target.position_x) - next_x)
            next_cone_width = max(12.0, max(self.attack_vertical_min, next_dy_up) * math.tan(math.radians(self.shot_cone_angle_degrees)))
            alignment = max(0.0, 1.0 - next_error / next_cone_width)
            improvement = max(-1.0, min(1.0, (current_error - next_error) / max(1.0, cone_width)))
            next_under_error = abs(next_dy_up - self.attack_under_distance)
            under_score = max(0.0, 1.0 - next_under_error / self.attack_under_band)
            under_improvement = max(-1.0, min(1.0, (current_under_error - next_under_error) / self.attack_under_band))
            priors[action] += scale * (
                0.45 * alignment
                + 0.20 * improvement
                + 0.25 * under_score
                + 0.10 * under_improvement
            )
            if next_error > current_error + 1e-6:
                priors[action] -= scale * 0.2
        return priors

    def _action_safety_priors(self, state, enemy_list):
        if state is None or len(state) < 20:
            return np.zeros(self.action_size, dtype=np.float32)

        x = float(state[18])
        y = float(state[19])
        action_delta = (
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        )
        penalties = np.zeros(self.action_size, dtype=np.float32)
        bullets = [
            bullet
            for enemy in enemy_list
            for bullet in enemy.bullets
            if getattr(bullet, "show", False)
        ]
        if not bullets:
            return penalties

        for action, (adx, ady) in enumerate(action_delta):
            action_penalty = 0.0
            for horizon in self.safety_lookahead_frames:
                px = min(self.wall_x, max(0.0, x + adx * self.action_step * horizon))
                py = min(self.wall_y, max(0.0, y + ady * self.action_step * horizon))
                horizon_penalty = 0.0
                for bullet in bullets:
                    bx = float(bullet.position_x)
                    by = float(bullet.position_y)
                    if abs(bx - px) > 180 or abs(by - py) > 180:
                        continue
                    vx, vy = self._bullet_velocity(bullet)
                    radius = float(getattr(bullet, "size", 0.0)) + self.player_radius
                    rel_x = px - bx
                    rel_y = py - by
                    current_clearance = math.hypot(rel_x, rel_y) - radius
                    speed = math.hypot(vx, vy)
                    if speed < 1e-6:
                        path_clearance = current_clearance
                        closing = 0.0
                    else:
                        closing = (rel_x * vx + rel_y * vy) / speed
                        path_clearance = abs(rel_x * vy - rel_y * vx) / speed - radius
                    clearance = min(current_clearance, path_clearance if closing > -radius else current_clearance)
                    if clearance < 0:
                        horizon_penalty += self.safety_penalty_scale * 2.5 / math.sqrt(horizon)
                    elif clearance < self.safety_margin:
                        danger = (self.safety_margin - clearance) / self.safety_margin
                        horizon_penalty += self.safety_penalty_scale * danger * danger / math.sqrt(horizon)
                action_penalty += horizon_penalty
            penalties[action] = -action_penalty
        return penalties

    @staticmethod
    def _bullet_velocity(bullet):
        count = getattr(bullet, "count", 0) or 5
        vx = (float(bullet.position_x) - float(getattr(bullet, "last_x", bullet.position_x))) / max(1, count)
        vy = (float(bullet.position_y) - float(getattr(bullet, "last_y", bullet.position_y))) / max(1, count)
        speed = math.hypot(vx, vy)
        max_speed = max(1.0, float(getattr(bullet, "speed", speed or 1.0)) * 2.0)
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale
        return vx, vy

    def learn(self, reward, next_full_state, action, done=False):
        if self.last_state is not None and self.last_action is not None:
            self.replay.push(self.last_state, self.last_action, reward, next_full_state, done)
        self.last_state = next_full_state
        self.last_action = action

        if len(self.replay) < self.batch_size or self.steps % self.train_every != 0:
            return

        for _ in range(self.gradient_steps):
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            grid = self._batch_grids(states)
            local_grid = self._batch_local_grids(states)
            vector = self._batch_vectors(states)
            next_grid = self._batch_grids(next_states)
            next_local_grid = self._batch_local_grids(next_states)
            next_vector = self._batch_vectors(next_states)
            action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            q = self.policy_net(grid, local_grid, vector).gather(1, action_tensor)
            with torch.no_grad():
                next_actions = self.policy_net(next_grid, next_local_grid, next_vector).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_grid, next_local_grid, next_vector).gather(1, next_actions)
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
            local_grid = self._batch_local_grids([state])
            vector = self._batch_vectors([state])
            return self.policy_net(grid, local_grid, vector).squeeze(0).detach().cpu().numpy()

    def _batch_grids(self, states):
        arr = np.stack([s.grid for s in states]).astype(np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _batch_local_grids(self, states):
        arr = np.stack([s.local_grid for s in states]).astype(np.float32)
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
        local_grid = self._encode_local_grid(enemy_list, x, y, wall_x, wall_y)

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
        base_reward = hit * 100.0 + hurt * -150.0
        aim_reward = (aim_center * 2.0 + (aim_left + aim_right) * 0.3 + aim_alignment * 1.0) * 0.4
        wall_reward = -wall_penalty * (1.0 if enemies_alive else 3.0)
        reward = (base_reward + aim_reward + wall_reward) * wall_reward_coef
        self.last_reward_components = {
            "hit": hit * 100.0,
            "hurt": hurt * -150.0,
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
        return GridState(grid=grid, local_grid=local_grid, vector=vector, features=features), reward

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

    def _encode_local_grid(self, enemy_list, x, y, wall_x, wall_y):
        grid = np.zeros(
            (self.local_grid_channels, self.local_grid_size, self.local_grid_size),
            dtype=np.float32,
        )

        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, "show", False):
                    continue
                bx, by = float(bullet.position_x), float(bullet.position_y)
                mapped = self._foveated_local_point(bx - x, by - y)
                if mapped is None:
                    continue
                gx, gy, radius = mapped
                vx, vy = self._bullet_velocity(bullet)
                speed = max(1e-6, math.hypot(vx, vy))
                toward = max(0.0, -((bx - x) * vx + (by - y) * vy) / (max(1.0, math.hypot(bx - x, by - y)) * speed))
                bullet_radius = max(1, radius + int(getattr(bullet, "size", 0) / 4))
                self._stamp(grid[0], gx, gy, bullet_radius, 1.0)
                self._stamp(grid[1], gx, gy, bullet_radius, max(-1.0, min(1.0, vx / 12.0)))
                self._stamp(grid[2], gx, gy, bullet_radius, max(-1.0, min(1.0, vy / 12.0)))
                self._stamp(grid[3], gx, gy, bullet_radius, toward)

        for enemy in enemy_list:
            if not getattr(enemy, "show", True):
                continue
            mapped = self._foveated_local_point(enemy.position_x - x, enemy.position_y - y)
            if mapped is None:
                continue
            gx, gy, radius = mapped
            hp_norm = min(1.0, max(0.05, getattr(enemy, "health", 1) / max(getattr(enemy, "full_health", 1), 1)))
            enemy_radius = max(1, radius + int(getattr(enemy, "size", 0) / 8))
            self._stamp(grid[4], gx, gy, enemy_radius, hp_norm)
            if getattr(enemy, "boss", False):
                self._stamp(grid[5], gx, gy, enemy_radius + 1, hp_norm)

        center = self.local_grid_size // 2
        self._stamp(grid[6], center, center, 1, 1.0)
        self._encode_local_wall_channel(grid[7], x, y, wall_x, wall_y)
        return grid

    def _foveated_local_point(self, rel_x, rel_y):
        radius = float(self.local_world_radius)
        if abs(rel_x) > radius or abs(rel_y) > radius:
            return None
        center = (self.local_grid_size - 1) / 2.0
        norm_x = abs(rel_x) / radius
        norm_y = abs(rel_y) / radius
        mapped_x = center + math.copysign(math.sqrt(norm_x) * center, rel_x) if rel_x else center
        mapped_y = center + math.copysign(math.sqrt(norm_y) * center, rel_y) if rel_y else center
        if not (0 <= mapped_x < self.local_grid_size and 0 <= mapped_y < self.local_grid_size):
            return None
        dist_norm = max(1e-3, math.hypot(rel_x, rel_y) / radius)
        local_radius = max(1, min(8, int(1.0 / math.sqrt(dist_norm))))
        return int(mapped_x), int(mapped_y), local_radius

    def _encode_local_wall_channel(self, channel, x, y, wall_x, wall_y):
        center = self.local_grid_size // 2
        radius = float(self.local_world_radius)
        for gy in range(self.local_grid_size):
            for gx in range(self.local_grid_size):
                nx = (gx - center) / max(1, center)
                ny = (gy - center) / max(1, center)
                rel_x = math.copysign(nx * nx * radius, nx)
                rel_y = math.copysign(ny * ny * radius, ny)
                wx = x + rel_x
                wy = y + rel_y
                dist = min(wx, wall_x - wx, wy, wall_y - wy)
                channel[gy, gx] = 1.0 - min(1.0, max(0.0, dist) / 80.0)

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
        xx, yy, _ = self._get_grid_cache(wall_x, wall_y)
        dx = xx - x
        dy_up = y - yy
        dist = np.sqrt(dx * dx + dy_up * dy_up)
        angle = np.abs(np.arctan2(dx, np.maximum(dy_up, 1e-3)))
        mask = (dy_up > 0) & (dist <= self.shot_range) & (angle <= cone)
        channel[mask] = 1.0 - np.minimum(1.0, angle[mask] / cone) * 0.5

    def _encode_wall_channel(self, channel):
        _, _, wall_channel = self._get_grid_cache(self.wall_x, self.wall_y)
        channel[:] = wall_channel

    def _get_grid_cache(self, wall_x, wall_y):
        key = (self.grid_width, self.grid_height, wall_x, wall_y)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        xs = (np.arange(self.grid_width, dtype=np.float32) + 0.5) * wall_x / self.grid_width
        ys = (np.arange(self.grid_height, dtype=np.float32) + 0.5) * wall_y / self.grid_height
        xx, yy = np.meshgrid(xs, ys)
        index_y, index_x = np.indices((self.grid_height, self.grid_width))
        left = index_x
        right = self.grid_width - 1 - index_x
        top = index_y
        bottom = self.grid_height - 1 - index_y
        dist = np.minimum(np.minimum(left, right), np.minimum(top, bottom)).astype(np.float32)
        wall_channel = 1.0 - np.minimum(1.0, dist / 10.0)
        cached = (xx, yy, wall_channel)
        self._grid_cache[key] = cached
        return cached

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
                    radius = d + bsize
                    dx = bx - zone_x
                    dy = by - zone_y
                    if dx * dx + dy * dy <= radius * radius:
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
