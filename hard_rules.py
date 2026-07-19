import math


class HardRulePolicy:
    def __init__(self, action_size=9, direction_range=10, warning_range=30):
        self.action_size = action_size
        self.direction_range = direction_range
        self.warning_range = warning_range
        self.enemy_init_hp = {}
        self.enemy_lock_id = None

    def filter_actions(self, full_state, enemy_list=None):
        bullet_counts = full_state[:9]
        x, y = full_state[18], full_state[19]
        wall_x, wall_y = int(full_state[22]), int(full_state[23])
        action_delta = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        ]

        self._update_enemy_hp(enemy_list)
        self._lock_enemy(enemy_list, wall_x, wall_y)
        warning_scores = self._calc_warning_scores(x, y, enemy_list)

        if any(cnt > 0 for cnt in bullet_counts):
            safe_dirs = [i for i, cnt in enumerate(bullet_counts) if cnt == 0]
            if safe_dirs:
                return safe_dirs

            edge_dirs = self._find_edge_dirs_with_least_bullet(enemy_list, x, y, wall_x, wall_y)
            if edge_dirs:
                return edge_dirs

            min_bullet = min(bullet_counts)
            return [i for i, cnt in enumerate(bullet_counts) if cnt == min_bullet]

        candidate_dirs = list(range(self.action_size))
        candidate_dirs = self._filter_toward_enemy(candidate_dirs, enemy_list, x, y, action_delta)
        candidate_dirs = self._filter_away_from_wall(candidate_dirs, x, y, wall_x, wall_y, action_delta)
        candidate_dirs = self._filter_by_warning(candidate_dirs, warning_scores)
        candidate_dirs = self._filter_toward_lower_center(candidate_dirs, x, y, wall_x, wall_y, action_delta)
        return candidate_dirs

    def _lock_enemy(self, enemy_list, wall_x, wall_y):
        if not enemy_list:
            self.enemy_lock_id = None
            return

        alive_enemies = [e for e in enemy_list if getattr(e, "show", True)]
        if not alive_enemies:
            self.enemy_lock_id = None
            return

        enemy_ids = [id(e) for e in alive_enemies]
        hp_by_id = {eid: self.enemy_init_hp.get(eid, 0) for eid in enemy_ids}
        if not hp_by_id:
            self.enemy_lock_id = None
            return

        max_hp = max(hp_by_id.values())
        candidates = [e for e in alive_enemies if self.enemy_init_hp.get(id(e), 0) == max_hp]
        if len(candidates) == 1:
            self.enemy_lock_id = id(candidates[0])
            return

        cx, cy = wall_x / 2, wall_y / 2
        locked = min(candidates, key=lambda e: (e.position_x - cx) ** 2 + (e.position_y - cy) ** 2)
        self.enemy_lock_id = id(locked)

    def _filter_toward_enemy(self, candidate_dirs, enemy_list, x, y, action_delta):
        if not enemy_list:
            return candidate_dirs

        enemies = [e for e in enemy_list if getattr(e, "show", True)]
        if not enemies:
            return candidate_dirs

        locked = next((e for e in enemies if id(e) == self.enemy_lock_id), None)
        targets = [locked] if locked is not None else enemies

        min_xdist = float("inf")
        best_x_dirs = []
        for i in candidate_dirs:
            dx, _ = action_delta[i]
            xx = x + dx * self.direction_range
            xdist = min(abs(xx - e.position_x) for e in targets)
            if xdist < min_xdist:
                min_xdist = xdist
                best_x_dirs = [i]
            elif xdist == min_xdist:
                best_x_dirs.append(i)

        best_y_dirs = []
        min_ydist = float("inf")
        for i in best_x_dirs:
            dx, dy = action_delta[i]
            xx = x + dx * self.direction_range
            yy = y + dy * self.direction_range
            for e in targets:
                if yy <= e.position_y + 1e-3:
                    ydist = abs(yy - e.position_y)
                    if ydist < min_ydist:
                        min_ydist = ydist
                        best_y_dirs = [i]
                    elif ydist == min_ydist:
                        best_y_dirs.append(i)

        return list(set(best_y_dirs)) if best_y_dirs else best_x_dirs

    def _filter_away_from_wall(self, candidate_dirs, x, y, wall_x, wall_y, action_delta):
        if len(candidate_dirs) <= 1:
            return candidate_dirs

        def wall_score(i):
            dx, dy = action_delta[i]
            xx = x + dx * self.direction_range
            yy = y + dy * self.direction_range
            return min(xx, wall_x - xx, yy, wall_y - yy)

        max_wall_score = max(wall_score(i) for i in candidate_dirs)
        return [i for i in candidate_dirs if abs(wall_score(i) - max_wall_score) < 1e-3]

    @staticmethod
    def _filter_by_warning(candidate_dirs, warning_scores):
        if len(candidate_dirs) <= 1:
            return candidate_dirs
        min_warning = min(warning_scores[i] for i in candidate_dirs)
        return [i for i in candidate_dirs if abs(warning_scores[i] - min_warning) < 1e-6]

    def _filter_toward_lower_center(self, candidate_dirs, x, y, wall_x, wall_y, action_delta):
        if len(candidate_dirs) <= 1:
            return candidate_dirs

        target_x = wall_x / 2
        target_y = wall_y * 0.8

        def center_score(i):
            dx, dy = action_delta[i]
            xx = x + dx * self.direction_range
            yy = y + dy * self.direction_range
            return math.hypot(xx - target_x, yy - target_y)

        min_center_score = min(center_score(i) for i in candidate_dirs)
        return [i for i in candidate_dirs if abs(center_score(i) - min_center_score) < 1e-3]

    def _calc_warning_scores(self, x, y, enemy_list):
        warning_radius = 300
        d = self.direction_range
        wr = self.warning_range
        diag = d / math.sqrt(2)
        danger_centers = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y),
        ]
        warning_scores = [0.0] * self.action_size

        if not enemy_list:
            return warning_scores

        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, "show", False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                bvx = getattr(bullet, "vx", 0.0)
                bvy = getattr(bullet, "vy", 0.0)
                bsize = getattr(bullet, "size", 0)
                speed = math.hypot(bvx, bvy)
                if speed < 1e-6:
                    speed = 1e-6
                if math.hypot(bx - x, by - y) > warning_radius + bsize:
                    continue
                for i, (cx, cy) in enumerate(danger_centers):
                    dist = math.hypot(bx - cx, by - cy)
                    if dist <= wr + bsize:
                        warning_scores[i] += (max(dist, 1.0) + bsize) / speed
        return warning_scores

    def _update_enemy_hp(self, enemy_list):
        if enemy_list is None:
            self.enemy_init_hp.clear()
            self.enemy_lock_id = None
            return

        for e in enemy_list:
            eid = id(e)
            if eid not in self.enemy_init_hp:
                self.enemy_init_hp[eid] = getattr(e, "health", 0)

        live_ids = {id(e) for e in enemy_list}
        for eid in [eid for eid in self.enemy_init_hp if eid not in live_ids]:
            del self.enemy_init_hp[eid]
        if self.enemy_lock_id is not None and self.enemy_lock_id not in live_ids:
            self.enemy_lock_id = None

    @staticmethod
    def _find_edge_dirs_with_least_bullet(enemy_list, x, y, wall_x, wall_y):
        if enemy_list is None:
            return []

        edge_width = 60
        edge_bullet_counts = [0, 0, 0, 0]
        for enemy in enemy_list:
            for bullet in enemy.bullets:
                if not getattr(bullet, "show", False):
                    continue
                bx, by = bullet.position_x, bullet.position_y
                if by < edge_width:
                    edge_bullet_counts[0] += 1
                if by > wall_y - edge_width:
                    edge_bullet_counts[1] += 1
                if bx < edge_width:
                    edge_bullet_counts[2] += 1
                if bx > wall_x - edge_width:
                    edge_bullet_counts[3] += 1

        min_count = min(edge_bullet_counts)
        preferred_edges = [i for i, count in enumerate(edge_bullet_counts) if count == min_count]
        dir_map = {
            0: [1],
            1: [5],
            2: [7],
            3: [3],
        }
        result = []
        for edge in preferred_edges:
            result.extend(dir_map[edge])
        return result
