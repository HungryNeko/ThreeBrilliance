import math
import numpy as np
import random
from collections import defaultdict

class STGAgent:
    def __init__(self):
        self.action_size = 9
        self.alpha = 0.05
        self.gamma = 0.98
        self.epsilon = 0.02
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.wall_x = 600
        self.wall_y = 900
        self.direction_range = 10
        self.warning_range = 30  # 新增：危险判定半径
        self.last_state = None
        self.last_action = None
        self.last_position = [0, 0]
        self.enemy_init_hp = dict()   # {enemy_id: init_hp}
        self.enemy_lock_id = None     # 当前锁定的敌人id

    def get_action(self, full_state, enemy_list=None):
        """
        1. 九宫格有弹幕时，强制避弹（无弹方向优先，否则弹幕最少的方向）。
        2. 否则，主动全屏攻击（优先锁定血量最多的敌人，血量相同则中心距离最近，锁定后只追此敌）。
        3. 如果靠墙/角且被弹幕压制（所有方向都有弹），则扫描全屏，往弹幕最少的边强制移动。
        4. 优先向风险小的方向移动（warning_range风险评估）。
        5. Q表微调。
        """
        bullet_counts = full_state[:9]
        x, y = full_state[18], full_state[19]
        wall_x, wall_y = int(full_state[22]), int(full_state[23])
        action_delta = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        ]

        # === 记录所有当前敌人和初始血量 ===
        self._update_enemy_hp(enemy_list)

        # === 优先锁定血量最多的敌人（如有多名则中心近的） ===
        locked_enemy = None
        if enemy_list:
            alive_enemies = [e for e in enemy_list if getattr(e, 'show', True)]
            if alive_enemies:
                enemy_id_list = [id(e) for e in alive_enemies]
                filtered_hp = {eid: self.enemy_init_hp.get(eid, 0) for eid in enemy_id_list}
                if filtered_hp:
                    max_hp = max(filtered_hp.values())
                    candidates = [e for e in alive_enemies if self.enemy_init_hp.get(id(e), 0) == max_hp]
                    if len(candidates) == 1:
                        locked_enemy = candidates[0]
                    else:
                        cx, cy = wall_x / 2, wall_y / 2
                        locked_enemy = min(
                            candidates,
                            key=lambda e: (e.position_x - cx) ** 2 + (e.position_y - cy) ** 2
                        )
                    self.enemy_lock_id = id(locked_enemy)
                else:
                    self.enemy_lock_id = None
            else:
                self.enemy_lock_id = None
        else:
            self.enemy_lock_id = None

        # ==== 新增：九宫格危险度分析 ====
        warning_scores = self._calc_warning_scores(x, y, enemy_list)

        # === 动作决策 ===
        # 1. 判断是否需要强制避弹
        if any(cnt > 0 for cnt in bullet_counts):
            safe_dirs = [i for i, cnt in enumerate(bullet_counts) if cnt == 0]
            if safe_dirs:
                candidate_dirs = safe_dirs
            else:
                edge_dirs = self._find_edge_dirs_with_least_bullet(enemy_list, x, y, wall_x, wall_y)
                if edge_dirs:
                    candidate_dirs = edge_dirs
                else:
                    min_bullet = min(bullet_counts)
                    candidate_dirs = [i for i, cnt in enumerate(bullet_counts) if cnt == min_bullet]
        else:
            candidate_dirs = list(range(self.action_size))
            if enemy_list and self.enemy_lock_id is not None:
                locked = None
                for e in enemy_list:
                    if id(e) == self.enemy_lock_id and getattr(e, 'show', True):
                        locked = e
                        break
                if locked is not None:
                    target_enemy = locked
                    min_xdist = float('inf')
                    best_x_dirs = []
                    for i in candidate_dirs:
                        dx, dy = action_delta[i]
                        xx = x + dx * self.direction_range
                        xdist = abs(xx - target_enemy.position_x)
                        if xdist < min_xdist:
                            min_xdist = xdist
                            best_x_dirs = [i]
                        elif xdist == min_xdist:
                            best_x_dirs.append(i)
                    best_y_dirs = []
                    min_ydist = float('inf')
                    for i in best_x_dirs:
                        dx, dy = action_delta[i]
                        xx = x + dx * self.direction_range
                        yy = y + dy * self.direction_range
                        if yy <= target_enemy.position_y + 1e-3:
                            ydist = abs(yy - target_enemy.position_y)
                            if ydist < min_ydist:
                                min_ydist = ydist
                                best_y_dirs = [i]
                            elif ydist == min_ydist:
                                best_y_dirs.append(i)
                    if best_y_dirs:
                        candidate_dirs = list(set(best_y_dirs))
                    else:
                        candidate_dirs = best_x_dirs
            elif enemy_list:
                enemies = [e for e in enemy_list if getattr(e, 'show', True)]
                if enemies:
                    min_xdist = float('inf')
                    best_x_dirs = []
                    for i in candidate_dirs:
                        dx, dy = action_delta[i]
                        xx = x + dx * self.direction_range
                        xdist_to_enemies = [abs(xx - e.position_x) for e in enemies]
                        xdist = min(xdist_to_enemies)
                        if xdist < min_xdist:
                            min_xdist = xdist
                            best_x_dirs = [i]
                        elif xdist == min_xdist:
                            best_x_dirs.append(i)
                    best_y_dirs = []
                    min_ydist = float('inf')
                    for i in best_x_dirs:
                        dx, dy = action_delta[i]
                        xx = x + dx * self.direction_range
                        yy = y + dy * self.direction_range
                        for e in enemies:
                            if yy <= e.position_y + 1e-3:
                                ydist = abs(yy - e.position_y)
                                if ydist < min_ydist:
                                    min_ydist = ydist
                                    best_y_dirs = [i]
                                elif ydist == min_ydist:
                                    best_y_dirs.append(i)
                    if best_y_dirs:
                        candidate_dirs = list(set(best_y_dirs))
                    else:
                        candidate_dirs = best_x_dirs

            # 3. 多个等价优先空旷（远离墙角/边缘）
            if len(candidate_dirs) > 1:
                def wall_score(i):
                    dx, dy = action_delta[i]
                    xx = x + dx * self.direction_range
                    yy = y + dy * self.direction_range
                    return min(xx, wall_x - xx, yy, wall_y - yy)
                max_wall_score = max(wall_score(i) for i in candidate_dirs)
                candidate_dirs = [i for i in candidate_dirs if abs(wall_score(i) - max_wall_score) < 1e-3]

            # ==== 新增：多个等价优先向风险较小方向 ====
            if len(candidate_dirs) > 1:
                min_warning = min([warning_scores[i] for i in candidate_dirs])
                warning_best = [i for i in candidate_dirs if abs(warning_scores[i] - min_warning) < 1e-6]
                candidate_dirs = warning_best

            # 4. 多个等价再回中下
            if len(candidate_dirs) > 1:
                target_x = wall_x / 2
                target_y = wall_y * 0.8
                def to_center_score(i):
                    dx, dy = action_delta[i]
                    xx = x + dx * self.direction_range
                    yy = y + dy * self.direction_range
                    return math.hypot(xx - target_x, yy - target_y)
                min_center_score = min([to_center_score(i) for i in candidate_dirs])
                candidate_dirs = [i for i in candidate_dirs if abs(to_center_score(i) - min_center_score) < 1e-3]

        # 5. Q表微调
        q_values = self.q_table[self._state_to_key(full_state)]
        if len(candidate_dirs) == 1:
            action = candidate_dirs[0]
        else:
            if random.random() < self.epsilon:
                action = random.choice(candidate_dirs)
            else:
                action = max(candidate_dirs, key=lambda i: q_values[i])
        return action

    def _calc_warning_scores(self, x, y, enemy_list):
        """
        计算以当前(x, y)为中心，半径300，分为9宫格（中心点为direction_range的九宫格），
        每个区域半径warning_range，落在该区域内的每颗子弹按 距离/速度 加权累加，作为风险值。
        """
        warning_radius = 300
        d = self.direction_range
        wr = self.warning_range
        diag = d / math.sqrt(2)
        danger_centers = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y)
        ]
        warning_scores = [0.0] * 9

        if not enemy_list:
            return warning_scores

        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, 'show', False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                bvx = getattr(bullet, 'vx', 0.0)
                bvy = getattr(bullet, 'vy', 0.0)
                bsize = getattr(bullet, 'size', 0)
                speed = math.hypot(bvx, bvy)
                if speed < 1e-6:
                    speed = 1e-6
                bullet_dist = math.hypot(bx - x, by - y)
                if bullet_dist > warning_radius + bsize:
                    continue
                for i, (cx, cy) in enumerate(danger_centers):
                    d2 = math.hypot(bx - cx, by - cy)
                    if d2 <= wr + bsize:
                        # 风险分数加权，距离越近、速度越快越危险
                        # 距离权重越小风险越高，速度越小风险越高
                        warning_scores[i] += (max(d2, 1.0) + bsize) / speed
        return warning_scores

    def _update_enemy_hp(self, enemy_list):
        if enemy_list is None:
            self.enemy_init_hp.clear()
            self.enemy_lock_id = None
            return
        for e in enemy_list:
            eid = id(e)
            if eid not in self.enemy_init_hp:
                self.enemy_init_hp[eid] = getattr(e, 'health', 0)
        live_ids = set(id(e) for e in enemy_list)
        to_del = [eid for eid in self.enemy_init_hp if eid not in live_ids]
        for eid in to_del:
            del self.enemy_init_hp[eid]
        if self.enemy_lock_id is not None and self.enemy_lock_id not in live_ids:
            self.enemy_lock_id = None

    def _find_edge_dirs_with_least_bullet(self, enemy_list, x, y, wall_x, wall_y):
        if enemy_list is None:
            return []
        edge_width = 60
        edge_bullet_counts = [0, 0, 0, 0]
        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, 'show', False): continue
                bx, by = bullet.position_x, bullet.position_y
                if by < edge_width:
                    edge_bullet_counts[0] += 1  # 上
                if by > wall_y - edge_width:
                    edge_bullet_counts[1] += 1  # 下
                if bx < edge_width:
                    edge_bullet_counts[2] += 1  # 左
                if bx > wall_x - edge_width:
                    edge_bullet_counts[3] += 1  # 右
        min_count = min(edge_bullet_counts)
        preferred_edges = [i for i, c in enumerate(edge_bullet_counts) if c == min_count]
        dir_map = {
            0: [1],    # 上：正上
            1: [5],    # 下：正下
            2: [7],    # 左：正左
            3: [3],    # 右：正右
        }
        result = []
        for edge in preferred_edges:
            result.extend(dir_map[edge])
        return result

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
            return tuple([0]*16)
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
        return threat_feat + (norm_x, norm_y, norm_ex, norm_ey, wall_penalty)

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
        reward = (base_reward - wall_penalty * (1. if enemies_alive else 3.0)) * wall_reward_coef

        full_state = (
            *bullet_inbox_counts,    # 0-8
            *threat_scores,          # 9-17
            x, y,                    # 18, 19
            norm_ex, norm_ey,        # 20, 21
            wall_x, wall_y           # 22, 23
        )
        self.last_position = [x, y]
        return full_state, reward

    def _get_direction_threat(self, x, y, wall_x, wall_y, enemy_list):
        move_step = 25
        max_lookahead = 500
        wall_repulse_dist = 100
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
                    if angle_diff > math.radians(10):
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