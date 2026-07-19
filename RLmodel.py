import math
import numpy as np
import random
from collections import defaultdict

from hard_rules import HardRulePolicy


ACTION_LABELS = (
    "left-up", "up", "right-up",
    "right", "right-down", "down",
    "left-down", "left", "stay",
)

class STGAgent:
    def __init__(self, use_hard_rules=False):
        self.action_size = 9
        self.alpha = 0.05
        self.gamma = 0.98
        self.epsilon = 0.02
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_prior_weight = 2.0
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.wall_x = 600
        self.wall_y = 900
        self.direction_range = 10
        self.warning_range = 30  # 新增：危险判定半径
        self.threat_move_step = 25
        self.max_lookahead = 500
        self.wall_repulse_dist = 100
        self.threat_angle_degrees = 10
        self.shot_range = 900
        self.shot_cone_angle_degrees = 14
        self.shot_center_angle_degrees = 4
        self.last_state = None
        self.last_action = None
        self.last_position = [0, 0]
        self.use_hard_rules = use_hard_rules
        self.hard_rules = HardRulePolicy(self.action_size, self.direction_range, self.warning_range)
        self.last_decision = {}

    def get_action(self, full_state, enemy_list=None):
        """
        Select the action from the actual RL policy.

        With hard rules disabled, the model can choose from all 9 actions.
        With hard rules enabled, hard_rules.py filters the candidate action set
        before epsilon-greedy Q selection.
        """
        state_key = self._state_to_key(full_state)
        q_values = self.q_table[state_key]
        if self.use_hard_rules:
            candidate_actions = self.hard_rules.filter_actions(full_state, enemy_list)
            policy_mode = "hard-rule-filtered-q"
        else:
            candidate_actions = list(range(self.action_size))
            policy_mode = "pure-q"

        if not candidate_actions:
            candidate_actions = list(range(self.action_size))

        action_priors = self._get_action_priors(full_state, enemy_list)
        adjusted_values = tuple(
            float(q_values[i]) + self.action_prior_weight * action_priors[i]
            for i in range(self.action_size)
        )
        explored = False
        if len(candidate_actions) == 1:
            action = candidate_actions[0]
        elif random.random() < self.epsilon:
            action = random.choice(candidate_actions)
            explored = True
        else:
            best_value = max(adjusted_values[i] for i in candidate_actions)
            best_actions = [
                i for i in candidate_actions
                if abs(adjusted_values[i] - best_value) < 1e-9
            ]
            action = random.choice(best_actions)

        self.last_decision = {
            "policy_mode": policy_mode,
            "hard_rules_enabled": self.use_hard_rules,
            "candidate_actions": tuple(candidate_actions),
            "action": action,
            "action_label": ACTION_LABELS[action],
            "epsilon": self.epsilon,
            "explored": explored,
            "q_values": tuple(float(q_values[i]) for i in range(self.action_size)),
            "action_priors": action_priors,
            "adjusted_values": adjusted_values,
            "state_key": state_key,
        }
        return action

    def _get_action_priors(self, full_state, enemy_list=None):
        if not enemy_list:
            return tuple(0.0 for _ in range(self.action_size))

        x, y = full_state[18], full_state[19]
        wall_x, wall_y = full_state[22], full_state[23]
        action_delta = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        ]
        priors = []
        for dx, dy in action_delta:
            next_x = max(0, min(wall_x, x + dx * self.direction_range))
            next_y = max(0, min(wall_y, y + dy * self.direction_range))
            left, center, right, alignment, dist_norm = self._get_shot_cone_features(
                next_x, next_y, wall_x, wall_y, enemy_list
            )
            aim_score = center * 2.0 + (left + right) * 0.6 + alignment * 2.0 - dist_norm
            wall_dist = min(next_x, wall_x - next_x, next_y, wall_y - next_y)
            wall_score = min(1.0, wall_dist / max(self.wall_repulse_dist, 1))
            priors.append(aim_score + wall_score * 0.5)
        return tuple(priors)

    def learn(self, reward, next_full_state, action):
        last_state_key = self._state_to_key(self.last_state) if self.last_state is not None else None
        next_state_key = self._state_to_key(next_full_state)
        if last_state_key is None:
            self.last_state = next_full_state
            self.last_action = action
            return
        current_q = self.q_table[last_state_key][self.last_action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[last_state_key][self.last_action] = new_q
        self.last_state = next_full_state
        self.last_action = action
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _state_to_key(self, state):
        if state is None:
            return tuple([0]*19)
        threat_scores = state[9:18]
        def quantize(val, bins=(0, 1, 5, 20, 100)):
            if val == 0: return 0.0
            elif val <= bins[1]: return 0.25
            elif val <= bins[2]: return 0.5
            elif val <= bins[3]: return 0.75
            else: return 1.0
        threat_feat = tuple(quantize(x) for x in threat_scores)
        x = state[18]
        y = state[19]
        wall_x = state[22]
        wall_y = state[23]
        def pos_norm(v, vmax):
            if vmax == 0: return 0.0
            frac = v / vmax
            if frac < 0.125: return 0.0
            elif frac < 0.375: return 0.25
            elif frac < 0.625: return 0.5
            elif frac < 0.875: return 0.75
            else: return 1.0
        norm_x = pos_norm(x, wall_x)
        norm_y = pos_norm(y, wall_y)
        norm_ex = float(state[20])
        norm_ey = float(state[21])
        dist_left = x
        dist_right = wall_x - x
        dist_top = y
        dist_bottom = wall_y - y
        min_dist_to_wall = min(dist_left, dist_right, dist_top, dist_bottom)
        if min_dist_to_wall < 50:
            wall_penalty = 1.0
        elif min_dist_to_wall < 100:
            wall_penalty = 0.66
        elif min_dist_to_wall < 200:
            wall_penalty = 0.33
        else:
            wall_penalty = 0.0
        aim_left = self._quantize_count(state[24]) if len(state) > 24 else 0.0
        aim_center = self._quantize_count(state[25]) if len(state) > 25 else 0.0
        aim_right = self._quantize_count(state[26]) if len(state) > 26 else 0.0
        aim_alignment = self._quantize_unit(state[27]) if len(state) > 27 else 0.0
        aim_dist = self._quantize_unit(state[28]) if len(state) > 28 else 1.0
        return threat_feat + (
            norm_x, norm_y, norm_ex, norm_ey, wall_penalty,
            aim_left, aim_center, aim_right, aim_alignment, aim_dist,
        )

    @staticmethod
    def _quantize_count(count):
        if count <= 0:
            return 0.0
        if count == 1:
            return 0.33
        if count == 2:
            return 0.66
        return 1.0

    @staticmethod
    def _quantize_unit(value):
        value = max(0.0, min(1.0, float(value)))
        if value < 0.125: return 0.0
        elif value < 0.375: return 0.25
        elif value < 0.625: return 0.5
        elif value < 0.875: return 0.75
        else: return 1.0

    def _process_game_state(self, enemy_list, player_pos, hit, hurt, wall_x, wall_y):
        self.wall_x = wall_x
        self.wall_y = wall_y
        x, y = player_pos

        d = self.direction_range
        diag = d / math.sqrt(2)
        directions = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y)
        ]
        bullet_inbox_counts = [0] * 9

        # 统计九宫格各方向有多少弹幕（包含子弹大小）
        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, 'show', False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                bsize = getattr(bullet, 'size', 0)
                for i, (zone_x, zone_y) in enumerate(directions):
                    dist = math.hypot(bx - zone_x, by - zone_y)
                    if dist <= d + bsize:
                        bullet_inbox_counts[i] += 1

        # 计算9方向威胁度
        threat_scores = self._get_direction_threat(x, y, wall_x, wall_y, enemy_list)

        # 归一化敌人位置
        enemies = [e for e in enemy_list if getattr(e, 'show', True)]
        if enemies:
            closest_enemy = min(enemies, key=lambda e: abs(e.position_x - x) + abs(e.position_y - y))
            rel_ex = (closest_enemy.position_x - x)
            rel_ey = (closest_enemy.position_y - y)
            rel_ex = max(-wall_x, min(wall_x, rel_ex))
            rel_ey = max(-wall_y, min(wall_y, rel_ey))
            rel_x_norm = (rel_ex / wall_x + 1) / 2
            rel_y_norm = (rel_ey / wall_y + 1) / 2
            def rel_quantize(frac):
                if frac < 0.125: return 0.0
                elif frac < 0.375: return 0.25
                elif frac < 0.625: return 0.5
                elif frac < 0.875: return 0.75
                else: return 1.0
            norm_ex = rel_quantize(rel_x_norm)
            norm_ey = rel_quantize(rel_y_norm)
        else:
            norm_ex = 0.5
            norm_ey = 0.5

        # 靠墙指数惩罚
        wall_penalty_dist = 100
        wall_punish_base = 1.0
        k = 15.0
        dist_left = x
        dist_right = wall_x - x
        dist_top = y
        dist_bottom = wall_y - y
        min_dist_to_wall = min(dist_left, dist_right, dist_top, dist_bottom)
        if min_dist_to_wall < wall_penalty_dist:
            wall_penalty = wall_punish_base * math.exp((wall_penalty_dist - min_dist_to_wall) / k)
        else:
            wall_penalty = 0.0
        enemies_alive = any(getattr(e, 'show', True) for e in enemy_list)
        if min_dist_to_wall > 200:
            wall_reward_coef = 1.0
        elif min_dist_to_wall > 100:
            wall_reward_coef = 0.6
        elif min_dist_to_wall > 50:
            wall_reward_coef = 0.4
        else:
            wall_reward_coef = 0.2

        base_reward = (
            hit * 100.0 +
            hurt * -100.0
        )
        aim_features = self._get_shot_cone_features(x, y, wall_x, wall_y, enemy_list)
        aim_left, aim_center, aim_right, aim_alignment, aim_dist = aim_features
        aim_reward = (aim_center * 2.0 + (aim_left + aim_right) * 0.6 + aim_alignment * 2.0) * 0.5
        wall_reward = -wall_penalty * (1. if enemies_alive else 3.0)
        reward = (base_reward + aim_reward + wall_reward) * wall_reward_coef
        self.last_reward_components = {
            "hit": hit * 100.0,
            "hurt": hurt * -100.0,
            "aim": aim_reward,
            "wall": wall_reward,
            "wall_coef": wall_reward_coef,
            "total": reward,
        }

        full_state = (
            *bullet_inbox_counts,    # 0-8
            *threat_scores,          # 9-17
            x, y,                    # 18, 19
            norm_ex, norm_ey,        # 20, 21
            wall_x, wall_y,          # 22, 23
            *aim_features            # 24-28
        )
        self.last_position = [x, y]
        return full_state, reward

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

        return (
            left_count,
            center_count,
            right_count,
            best_alignment,
            nearest_dist_norm,
        )

    def _get_direction_threat(self, x, y, wall_x, wall_y, enemy_list):
        move_step = self.threat_move_step
        max_lookahead = self.max_lookahead
        wall_repulse_dist = self.wall_repulse_dist
        wall_punish_base = 20.0
        k = 15.0
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
                    if not getattr(bullet, 'show', False): continue
                    bx, by = bullet.position_x, bullet.position_y
                    bvx = getattr(bullet, 'vx', 0.0)
                    bvy = getattr(bullet, 'vy', 0.0)
                    speed = math.hypot(bvx, bvy)
                    if speed < 1e-3: continue
                    bullet_angle = math.atan2(bvy, bvx)
                    angle_diff = abs(self._angle_diff(bullet_angle, dir_angle))
                    if angle_diff > math.radians(self.threat_angle_degrees):
                        continue
                    dist = math.hypot(bx - x, by - y)
                    if dist > max_lookahead:
                        continue
                    score += (max_lookahead - dist) / speed
            new_x = x + dx * move_step
            new_y = y + dy * move_step
            dist_left = new_x
            dist_right = wall_x - new_x
            dist_top = new_y
            dist_bottom = wall_y - new_y
            min_dist_to_wall = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist_to_wall < wall_repulse_dist:
                wall_penalty = wall_punish_base * math.exp((wall_repulse_dist - min_dist_to_wall) / k)
                score += wall_penalty
            threat_scores[dir_idx] = score
        return threat_scores

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), q_table)
