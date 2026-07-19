import math
import argparse
import json
import os
import random
import sys
import time

if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from sympy.physics.units import action

import character
import img
from drl_agent import DRLAgent as STGAgent


class Train:
    def __init__(
            self,
            headless=False,
            render_every=5,
            log_every=1000,
            checkpoint="current_best_dqn.pth",
            log_file="training_log.jsonl",
            load_training_state=True):
        self.boss_exist = False
        self.last_time_hit=0
        self.rand=0
        self.countfps=0
        self.last_time_hurt = 0
        self._last_processed_state= None
        self._last_action = None
        self.action=8
        self.headless = headless
        self.checkpoint = checkpoint
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        self.pygame = pygame
        self.pygame.init()
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 900
        self.visual = not self.headless
        if self.visual:
            self.ui_scale = self._initial_ui_scale()
            self.display_size = self._scaled_size(self.ui_scale)
            self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
            self.window = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT)).convert()
            self.pygame.display.set_caption("ThreeBrilliance")
        else:
            self.ui_scale = 1.0
            self.display_size = (0, 0)
            self.display = None
            self.window = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.overlay_mode = 2
        self.debug_font = pygame.font.SysFont(None, 18) if self.visual else None
        self.debug_font_small = pygame.font.SysFont(None, 16) if self.visual else None
        self.fast_training = True
        self.frame_cap = 0
        self.render_every = render_every
        self.log_every = log_every
        self.audio_enabled = False
        self._image_cache = {}
        self.player1 = character.player(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2,
                                        10, 5, True,
                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT - 5], 100, (20, 255, 255))
        self.clock = pygame.time.Clock()
        self.enemy_list = []
        self.curtime= time.time()
        self.count = 0
        self.enemy_type = 0
        self.learncount=0
        self.learntimes=0
        self.stage=0
        self.stage_kill=0
        self.stage_kill_boss=0
        self.best_stage = 0
        self.best_stage_kill = 0
        self.best_stage_kill_boss = 0
        self.best_avg_reward = float("-inf")
        self.episode = 0
        self.death_resets = 0
        self.boss_stall_resets = 0
        self.last_terminal_reason = ""
        self.boss_no_hit_frames = 0
        self.boss_low_hp_no_hit_frames = 0
        self.boss_stall_hp_pct = 0.35
        self.boss_stall_no_hit_limit = 2400
        self.boss_stall_low_hp_no_hit_limit = 1200
        self.start_time = time.time()
        self.total_reward = 0.0
        self.reward_steps = 0
        self.last_reward = 0.0
        self.window_reward = 0.0
        self.total_hit_power = 0.0
        self.total_hurt_power = 0.0
        self.total_shot_power = 0.0
        self.window_hit_power = 0.0
        self.window_hurt_power = 0.0
        self.window_shot_power = 0.0
        self.last_log_count = 0
        self.last_log_time = self.start_time
        self._last_status_width = 0
        self._had_enemies_for_reward = False
        self.boss_alive_frames = 0
        self.last_boss_time_penalty = 0.0
        self.total_boss_time_penalty = 0.0
        self.log_file_path = log_file
        self.log_file = open(log_file, "a", buffering=1) if log_file else None
        self.training_state_path = f"{checkpoint}.train_state.json" if checkpoint else None
        if load_training_state:
            self._load_training_state()

        # 资源加载
        self.spritesheet = img.load_character_spritesheet("src/img_1.png", 4, 3, 50, 50) if self.visual else []
        self.volume = .5 if self.audio_enabled else 0.0
        if self.audio_enabled:
            self.close_sound = character.safe_sound("src/东方原作音效/绀长擦弹.wav", self.volume)
            self.hit_sound = character.safe_sound("src/东方原作音效/莎莎火箭弹命中.wav", self.volume)
            self.crash_sound = character.safe_sound("src/东方原作音效/击破boss.wav", self.volume)
            self.sound = character.safe_sound("src/th15_13.mp3", self.volume)
            self.channel_sound = character.safe_channel(0)
            self.channel_hit = character.safe_channel(1)
            self.channel_close = character.safe_channel(2)
            self.channel_crash = character.safe_channel(3)
            self.sound.set_volume(self.volume)
            self.hit_sound.set_volume(self.volume)
            self.close_sound.set_volume(self.volume)
            self.crash_sound.set_volume(self.volume)
        else:
            self.close_sound = None
            self.hit_sound = None
            self.crash_sound = None
            self.sound = None
            self.channel_sound = None
            self.channel_hit = None
            self.channel_close = None
            self.channel_crash = None

        self.lastbosstime=0
        self.player1.level=4

        # 可调参数 for enemytype4
        self.enemytype4_cfg = {
            "enemy0_num": 2,
            "enemy1_num": 2,
            "boss0_num": 1,
            "enemy0_freq": 200,
            "enemy1_freq": 300,
            "boss0_unique": True, # 只允许同时存在一个boss
        }

    def new_enemy(self, i=0):

        if i == 0:
            if self.count % 50 == 0:
                self.enemy_list.append(character.enemy0(random.uniform(0, self.WINDOW_WIDTH), 0, 1.5,
                                                        10, True,
                                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 200, (255, 0, 0), 10))
        if i == 1:
            if self.count % 70 == 0:
                self.enemy_list.append(character.enemy1(random.uniform(0, self.WINDOW_WIDTH), 0, 0.5,
                                                        30, True,
                                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 500, (255, 0, 0), 10))
        if i == 3:
            if self.enemy_list == [] and self.stage_kill_boss<1:

                self.enemy_list.append(character.boss0(self.WINDOW_WIDTH // 2, 0, 0.5,
                                                       60, True,
                                                       [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 10000, (255, 0, 0), 1, volume=self.volume))
        if i == 4:
            # 4类型为 0/1/3 的结合，不新建类，而是调用各自生成
            # enemy0
            if self.enemytype4_cfg["enemy0_num"] > 0:
                if self.count % self.enemytype4_cfg["enemy0_freq"] == 0:
                    for _ in range(self.enemytype4_cfg["enemy0_num"]):
                        self.enemy_list.append(character.enemy0(random.uniform(0, self.WINDOW_WIDTH), 0, 1.5,
                                                                10, True,
                                                                [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 200, (255, 0, 0), 100))
            # enemy1
            if self.enemytype4_cfg["enemy1_num"] > 0:
                if self.count % self.enemytype4_cfg["enemy1_freq"] == 0:
                    for _ in range(self.enemytype4_cfg["enemy1_num"]):
                        self.enemy_list.append(character.enemy1(random.uniform(0, self.WINDOW_WIDTH), 0, 0.5,
                                                                30, True,
                                                                [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 500, (255, 0, 0), 100))
            # boss0
            if self.enemytype4_cfg["boss0_num"] > 0 and self.enemytype4_cfg["boss0_unique"]:

                if not self.boss_exist:
                    if self.lastbosstime==1 and self.stage_kill_boss<1:
                        self.lastbosstime=2
                        for _ in range(self.enemytype4_cfg["boss0_num"]):
                            self.enemy_list.append(character.boss0(self.WINDOW_WIDTH // 2, 0, 0.5,
                                                                   60, True,
                                                                   [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 20000, (255, 0, 0), 4, volume=self.volume))
                    else:
                        self.lastbosstime%= 300
                        self.lastbosstime+=1
        if i==5:
            if self.enemy_list==[] and self.enemytype4_cfg["boss0_num"] > 0 and self.enemytype4_cfg["boss0_unique"]:

                if not self.boss_exist:

                    if self.stage_kill_boss<1:
                        for _ in range(self.enemytype4_cfg["boss0_num"]):
                            self.enemy_list.append(character.boss1(self.WINDOW_WIDTH // 2, 0, 0.5,
                                                                   60, True,
                                                                   [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT],
                                                                   30000, (255, 0, 0), 4, volume=self.volume))
                    else:
                        self.lastbosstime %= 300
                        self.lastbosstime += 1

    def show_heath(self, character, good=False):
        if good:
            pygame.draw.rect(self.window, (0, 255, 0), (0, self.WINDOW_HEIGHT - 5, self.WINDOW_WIDTH * character.health / character.full_health, 5))
        else:
            rect = pygame.Rect(character.position_x - (character.size + 100) / 2, character.position_y - (character.size + 100) / 2, character.size + 100, character.size + 100)
            pygame.draw.arc(self.window, (255, 100, 0), rect, math.radians(+90), math.radians(360 * character.health / character.full_health + 90), 5)

    def draw_image(self, image_path, x, y, size):
        size = size * math.sqrt(2) / 2
        cache_key = (image_path, int(size))
        image = self._image_cache.get(cache_key)
        if image is None:
            try:
                image = pygame.image.load(image_path)
                image = pygame.transform.scale(image, (int(size), int(size)))
            except (FileNotFoundError, pygame.error):
                image = self._make_fallback_boss_image(int(size))
            self._image_cache[cache_key] = image
        image_rect = image.get_rect(center=(x, y))
        self.window.blit(image, image_rect)

    def _make_fallback_boss_image(self, size):
        size = max(24, int(size))
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        radius = max(8, int(size * 0.32))
        pygame.draw.circle(surface, (220, 60, 90), (center, center), radius)
        pygame.draw.circle(surface, (255, 210, 80), (center, center), radius, 3)
        pygame.draw.line(surface, (255, 255, 255), (center - radius, center), (center + radius, center), 2)
        pygame.draw.line(surface, (255, 255, 255), (center, center - radius), (center, center + radius), 2)
        return surface

    def _initial_ui_scale(self):
        info = pygame.display.Info()
        if info.current_w <= 0 or info.current_h <= 0:
            return 0.8
        width_fit = info.current_w * 0.9 / self.WINDOW_WIDTH
        height_fit = info.current_h * 0.82 / self.WINDOW_HEIGHT
        return max(0.35, min(1.0, width_fit, height_fit))

    def _scaled_size(self, scale):
        return (max(1, int(self.WINDOW_WIDTH * scale)), max(1, int(self.WINDOW_HEIGHT * scale)))

    def _set_display_size(self, size):
        width = max(240, int(size[0]))
        height = max(360, int(size[1]))
        self.display_size = (width, height)
        self.ui_scale = min(width / self.WINDOW_WIDTH, height / self.WINDOW_HEIGHT)
        self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)

    def _change_ui_scale(self, delta):
        self.ui_scale = max(0.35, min(1.5, self.ui_scale + delta))
        self.display_size = self._scaled_size(self.ui_scale)
        self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)

    def _present(self):
        if self.display is None:
            return
        scaled_size = self._scaled_size(self.ui_scale)
        scaled_frame = pygame.transform.smoothscale(self.window, scaled_size)
        self.display.fill((0, 0, 0))
        offset_x = (self.display_size[0] - scaled_size[0]) // 2
        offset_y = (self.display_size[1] - scaled_size[1]) // 2
        self.display.blit(scaled_frame, (offset_x, offset_y))
        pygame.display.update()

    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.VIDEORESIZE:
            self._set_display_size((event.w, event.h))
            return
        if event.type != pygame.KEYDOWN:
            return

        if event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._change_ui_scale(0.1)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._change_ui_scale(-0.1)
        elif event.key in (pygame.K_0, pygame.K_KP0):
            self.ui_scale = self._initial_ui_scale()
            self.display_size = self._scaled_size(self.ui_scale)
            self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
        elif event.key == pygame.K_r:
            self.overlay_mode = (self.overlay_mode + 1) % 3

    def _should_render_frame(self):
        if not self.visual:
            return False
        if not self.fast_training:
            return True
        return self.count % self.render_every == 0

    def _collect_metrics(self, agent):
        now = time.time()
        elapsed = max(1e-6, now - self.start_time)
        window_elapsed = max(1e-6, now - self.last_log_time)
        sps = (self.count - self.last_log_count) / window_elapsed
        accuracy = self.total_hit_power / max(1.0, self.total_shot_power)
        window_accuracy = self.window_hit_power / max(1.0, self.window_shot_power)
        boss_pct = self._boss_hp_pct()
        decision = getattr(agent, "last_decision", {})
        loss = decision.get("loss")
        avg_reward = self.total_reward / max(1, self.reward_steps)
        if (self.stage, self.stage_kill, self.stage_kill_boss) > (
            self.best_stage,
            self.best_stage_kill,
            self.best_stage_kill_boss,
        ):
            self.best_stage = self.stage
            self.best_stage_kill = self.stage_kill
            self.best_stage_kill_boss = self.stage_kill_boss
        self.best_avg_reward = max(self.best_avg_reward, avg_reward)
        progress = f"{self.stage}.{self.stage_kill}/{self.stage_kill_boss}"
        best_progress = f"{self.best_stage}.{self.best_stage_kill}/{self.best_stage_kill_boss}"
        return {
            "step": self.count,
            "elapsed_sec": elapsed,
            "sps": sps,
            "device": str(decision.get("device", getattr(agent, "device", "unknown"))),
            "progress": progress,
            "best_progress": best_progress,
            "stage": self.stage,
            "stage_kill": self.stage_kill,
            "stage_boss_kill": self.stage_kill_boss,
            "episode": self.episode,
            "death_resets": self.death_resets,
            "boss_stall_resets": self.boss_stall_resets,
            "last_terminal_reason": self.last_terminal_reason,
            "player_hp": self.player1.health,
            "player_full_hp": self.player1.full_health,
            "boss_hp_pct": boss_pct,
            "accuracy": accuracy,
            "window_accuracy": window_accuracy,
            "hit_power": self.total_hit_power,
            "hurt_power": self.total_hurt_power,
            "shot_power": self.total_shot_power,
            "window_hit_power": self.window_hit_power,
            "window_hurt_power": self.window_hurt_power,
            "window_shot_power": self.window_shot_power,
            "reward": self.last_reward,
            "avg_reward": avg_reward,
            "best_avg_reward": self.best_avg_reward,
            "window_reward": self.window_reward,
            "epsilon": getattr(agent, "epsilon", 0.0),
            "loss": loss,
            "replay_size": decision.get("replay_size", 0),
            "updates": decision.get("updates", 0),
            "action": decision.get("action", self.action),
            "action_label": decision.get("action_label", ""),
            "boss_alive_frames": self.boss_alive_frames,
            "boss_no_hit_frames": self.boss_no_hit_frames,
            "boss_low_hp_no_hit_frames": self.boss_low_hp_no_hit_frames,
            "boss_time_penalty": self.last_boss_time_penalty,
            "total_boss_time_penalty": self.total_boss_time_penalty,
        }

    def _write_log_record(self, metrics):
        if self.log_file is None:
            return
        self.log_file.write(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _save_training_state(self):
        if not self.training_state_path:
            return
        state = {
            "total_reward": self.total_reward,
            "reward_steps": self.reward_steps,
            "best_stage": self.best_stage,
            "best_stage_kill": self.best_stage_kill,
            "best_stage_kill_boss": self.best_stage_kill_boss,
            "best_avg_reward": self.best_avg_reward,
            "episode": self.episode,
            "death_resets": self.death_resets,
            "boss_stall_resets": self.boss_stall_resets,
        }
        with open(self.training_state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))

    def _load_training_state(self):
        if not self.training_state_path or not os.path.exists(self.training_state_path):
            return
        try:
            with open(self.training_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self.total_reward = float(state.get("total_reward", self.total_reward))
        self.reward_steps = int(state.get("reward_steps", self.reward_steps))
        self.best_stage = int(state.get("best_stage", self.best_stage))
        self.best_stage_kill = int(state.get("best_stage_kill", self.best_stage_kill))
        self.best_stage_kill_boss = int(state.get("best_stage_kill_boss", self.best_stage_kill_boss))
        self.best_avg_reward = float(state.get("best_avg_reward", self.best_avg_reward))
        self.episode = int(state.get("episode", self.episode))
        self.death_resets = int(state.get("death_resets", self.death_resets))
        self.boss_stall_resets = int(state.get("boss_stall_resets", self.boss_stall_resets))

    def _reset_log_window(self):
        self.window_reward = 0.0
        self.window_hit_power = 0.0
        self.window_hurt_power = 0.0
        self.window_shot_power = 0.0

    def _update_boss_time_penalty(self):
        boss_alive = any(
            getattr(enemy, "boss", False) and getattr(enemy, "show", True)
            for enemy in self.enemy_list
        )
        if not boss_alive:
            self.boss_alive_frames = 0
            self.last_boss_time_penalty = 0.0
            return 0.0

        self.boss_alive_frames += 1
        penalty = -min(50.0, 0.002 * self.boss_alive_frames)
        self.last_boss_time_penalty = penalty
        self.total_boss_time_penalty += penalty
        return penalty

    def _boss_hp_pct(self):
        boss_hp = 0.0
        boss_max_hp = 0.0
        for enemy in self.enemy_list:
            if getattr(enemy, "boss", False) and getattr(enemy, "show", True):
                boss_hp += max(0.0, getattr(enemy, "health", 0.0))
                boss_max_hp += max(1.0, getattr(enemy, "full_health", 1.0))
        return boss_hp / boss_max_hp if boss_max_hp else 0.0

    def _update_player_homing_targets(self):
        targets = [enemy for enemy in self.enemy_list if getattr(enemy, "show", True)]
        if not targets:
            return
        bosses = [enemy for enemy in targets if getattr(enemy, "boss", False)]
        for bullet in self.player1.bullets:
            if getattr(bullet, "move_func", None) != "homing_curve" or not getattr(bullet, "show", False):
                continue
            candidates = [
                enemy
                for enemy in (bosses or targets)
                if enemy.position_y <= bullet.position_y + 120
            ]
            if not candidates:
                candidates = bosses or targets
            target = min(
                candidates,
                key=lambda enemy: abs(enemy.position_x - bullet.position_x) + abs(enemy.position_y - bullet.position_y),
            )
            bullet.homing_target_x = target.position_x
            bullet.homing_target_y = target.position_y

    def _update_boss_stall_counter(self):
        boss_pct = self._boss_hp_pct()
        boss_alive = boss_pct > 0.0
        if not boss_alive:
            self.boss_no_hit_frames = 0
            self.boss_low_hp_no_hit_frames = 0
            return False
        if self.last_time_hit > 0:
            self.boss_no_hit_frames = 0
            self.boss_low_hp_no_hit_frames = 0
            return False
        self.boss_no_hit_frames += 1
        if boss_pct > self.boss_stall_hp_pct:
            self.boss_low_hp_no_hit_frames = 0
            return self.boss_no_hit_frames >= self.boss_stall_no_hit_limit
        self.boss_low_hp_no_hit_frames += 1
        return self.boss_low_hp_no_hit_frames >= self.boss_stall_low_hp_no_hit_limit

    def _enemy_type_for_stage(self, stage):
        return {
            0: 0,
            1: 1,
            2: 3,
            3: 4,
            4: 5,
        }.get(stage, 0)

    def _reset_episode(self, reason):
        reset_stage = self.stage
        self.episode += 1
        if reason == "death":
            self.death_resets += 1
        elif reason == "boss_stall":
            self.boss_stall_resets += 1
        self.last_terminal_reason = reason
        self.player1 = character.player(
            self.WINDOW_WIDTH // 2,
            self.WINDOW_HEIGHT // 2,
            10,
            5,
            True,
            [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT - 5],
            100,
            (20, 255, 255),
        )
        self.player1.level = 4
        self.enemy_list = []
        self.stage = reset_stage
        self.stage_kill = 0
        self.stage_kill_boss = 0
        self.enemy_type = self._enemy_type_for_stage(self.stage)
        self.boss_exist = False
        self.lastbosstime = 0
        self.boss_alive_frames = 0
        self.last_boss_time_penalty = 0.0
        self.boss_no_hit_frames = 0
        self.boss_low_hp_no_hit_frames = 0
        self._had_enemies_for_reward = False
        self._last_processed_state = None
        self._last_action = None
        self.action = 8
        self.last_time_hit = 0
        self.last_time_hurt = 0

    def _learn_terminal_transition(self, agent, reason, penalty):
        if self._last_processed_state is None or self._last_action is None:
            return
        terminal_state, reward = agent._process_game_state(
            self.enemy_list,
            [self.player1.position_x, self.player1.position_y],
            self.last_time_hit,
            self.last_time_hurt,
            self.WINDOW_WIDTH,
            self.WINDOW_HEIGHT,
        )
        reward += penalty
        self.last_reward = reward
        self.total_reward += reward
        self.reward_steps += 1
        self.window_reward += reward
        agent.learn(reward, terminal_state, self._last_action, done=True)
        agent.last_state = None
        agent.last_action = None

    def _handle_terminal_state(self, agent):
        if self.player1.health <= 0:
            self._learn_terminal_transition(agent, "death", -8000.0)
            self._reset_episode("death")
            return True
        if self._update_boss_stall_counter():
            self._learn_terminal_transition(agent, "boss_stall", -6000.0)
            self._reset_episode("boss_stall")
            return True
        return False

    def _print_text_status(self, agent):
        metrics = self._collect_metrics(agent)
        status = " | ".join(
            [
                f"step={metrics['step']}",
                f"prog={metrics['progress']}",
                f"best={metrics['best_progress']}",
                f"avg={metrics['avg_reward']:.1f}",
                f"best_avg={metrics['best_avg_reward']:.1f}",
                f"reset={metrics['death_resets']}/{metrics['boss_stall_resets']}",
            ]
        )
        padding = max(0, self._last_status_width - len(status))
        print("\r" + status + (" " * padding), end="", flush=True)
        self._last_status_width = len(status)
        self._write_log_record(metrics)
        self._save_training_state()
        self._reset_log_window()
        self.last_log_count = self.count
        self.last_log_time = time.time()

    def action1(self):
        self.learncount+=1
        self.learncount%= 10000
        if self.learntimes%1==0:
            if self.learncount==1:
                #self.learntimes+=1
                agent.save(self.checkpoint)  # 保存模型状态
                self._save_training_state()
            # 1. 如果存在上一次的状态和动作，先进行学习
            current_state = None
            if self._last_processed_state is not None and self._last_action is not None:
                # 获取当前状态
                current_state, reward = agent._process_game_state(
                    self.enemy_list,
                    [self.player1.position_x, self.player1.position_y],
                    self.last_time_hit,
                    self.last_time_hurt,
                    self.WINDOW_WIDTH,
                    self.WINDOW_HEIGHT
                )
                enemies_alive_for_reward = len(self.enemy_list) > 0
                if self._had_enemies_for_reward and not enemies_alive_for_reward:
                    reward += 100  # 消灭所有敌人奖励
                self._had_enemies_for_reward = enemies_alive_for_reward
                reward += self._update_boss_time_penalty()
                self.last_reward = reward
                self.total_reward += reward
                self.reward_steps += 1
                self.window_reward += reward

                # 进行Q-learning更新
                agent.learn(reward, current_state, self._last_action)

            # 2. 处理当前状态并获取新动作
            if current_state is None:
                processed_state, current_reward = agent._process_game_state(
                    self.enemy_list,
                    [self.player1.position_x, self.player1.position_y],
                    self.last_time_hit,
                    self.last_time_hurt,
                    self.WINDOW_WIDTH,
                    self.WINDOW_HEIGHT
                )
            else:
                processed_state = current_state
            #print(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

            # 3. 获取动作
            action = agent.get_action(processed_state,self.enemy_list)

            # 4. 存储当前状态和动作
            self._last_processed_state = processed_state
            self._last_action = action
            self.action = action
            # if self.player1.level<4:
            #     self.player1.level+=(self.last_time_hit*0.001)*(4-self.player1.level)
            # if self.player1.level>0:
            #     self.player1.level-= self.last_time_hurt*(4-self.player1.level)*0.5
            # 5. 重置命中/受伤计数器
            self.last_time_hit = 0
            self.last_time_hurt = 0
            #print(action)

    def run(self):
        if self.visual and self.audio_enabled:
            #self.channel_sound.play(self.sound)
            pass
        #time.sleep(10)
        while True:
            self.action1()
            direction = 0
            self.count += 1
            if self.count == sys.maxsize:
                self.count = 0
            if self.visual:
                for event in pygame.event.get():
                    self._handle_event(event)
            shift = 1

            bullet_count_before_shoot = len(self.player1.bullets)
            self.player1.shoot(shift)
            new_player_bullets = self.player1.bullets[bullet_count_before_shoot:]
            shot_power = sum(getattr(bullet, "reward_power", getattr(bullet, "power", 0.0)) for bullet in new_player_bullets)
            self.total_shot_power += shot_power
            self.window_shot_power += shot_power
            if self.action == 0:  # 左上
                self.player1.move_x(-shift)
                self.player1.move_y(-shift)
            elif self.action == 1:  # 上
                self.player1.move_y(-shift)
            elif self.action == 2:  # 右上
                self.player1.move_x(shift)
                self.player1.move_y(-shift)
            elif self.action == 3:  # 右
                self.player1.move_x(shift)
            elif self.action == 4:  # 右下
                self.player1.move_x(shift)
                self.player1.move_y(shift)
            elif self.action == 5:  # 下
                self.player1.move_y(shift)
            elif self.action == 6:  # 左下
                self.player1.move_x(-shift)
                self.player1.move_y(shift)
            elif self.action == 7:  # 左
                self.player1.move_x(-shift)
            elif self.action == 8:  # 不动
                pass


            # AI/训练接口

            self.last_time_hit = 0
            self.last_time_hurt = 0

            should_render = self._should_render_frame()
            if should_render:
                self.window.fill((0, 0, 0))
            if self.stage==3 and self.boss_exist==False and self.stage_kill_boss>0:
                self.enemy_list=[]
            if self.stage==0 and self.stage_kill>10 and not self.boss_exist:
                self.stage=1
                self.enemy_type=1
                self.stage_kill_boss = 0
                self.stage_kill=0
            elif self.stage==1 and self.stage_kill>10 and not self.boss_exist:
                if self.audio_enabled:
                    self.sound.play(-1)
                self.stage=2
                self.stage_kill=0
                self.stage_kill_boss = 0
                self.enemy_type=3
            elif self.stage == 2 and self.stage_kill_boss > 0 and not self.boss_exist:
                self.stage = 3
                self.stage_kill = 0
                self.stage_kill_boss = 0
                self.enemy_type = 4
            elif self.stage==3 and self.stage_kill_boss > 0 and not self.boss_exist :
                self.stage= 4
                self.stage_kill=0
                self.stage_kill_boss = 0
                self.enemy_type=5
            elif self.stage==4 and self.stage_kill_boss>0 and not self.boss_exist :
                self.stage=0
                self.stage_kill_boss=0
                self.stage_kill=0
                self.enemy_type=0
            self.boss_exist = any(getattr(e, "boss", False) and e.show for e in self.enemy_list)
            self.new_enemy(self.enemy_type)
            if len(self.enemy_list) > 0:
                remove_en = []
                for i, e in enumerate(self.enemy_list):
                    if e.show:
                        if e.boss:
                            if self.stage==5:
                                e.position_x,e.position_y=self.move_randomly(e.position_x,e.position_y,e.speed*1.5)
                            if should_render:
                                self.show_heath(e, good=False)
                                self.draw_image('src/img.png', e.position_x, e.position_y - 25, e.size + 100)
                        elif should_render:
                            pygame.draw.circle(self.window, e.color, (e.position_x, e.position_y), e.size)
            self._update_player_homing_targets()
            self.player1.update_bullets()

            for i1, i in enumerate(self.player1.bullets):
                if should_render and self.count % 2 == 0:
                    x, y = i.before(0, 0.2)
                    color_1 = tuple(max(0, value - 50) for value in i.color)
                    pygame.draw.circle(self.window, color_1, (x, y), i.size)
                if should_render and self.count % 3 == 0:
                    x, y = i.before(0, 0.5)
                    color_2 = tuple(max(0, value - 50) for value in i.color)
                    pygame.draw.circle(self.window, color_2, (x, y), i.size)
            #hit检测
            for i1, i in enumerate(self.player1.bullets):
                if should_render:
                    pygame.draw.circle(self.window, i.color, (i.position_x, i.position_y), i.size)
                for e in self.enemy_list:
                    hit_radius = e.size + i.size
                    dx = i.position_x - e.position_x
                    dy = e.position_y - i.position_y
                    if dx * dx + dy * dy < hit_radius * hit_radius and e.show:
                        e.change_health(-i.power)
                        reward_power = getattr(i, "reward_power", i.power)
                        if e.health<=0:
                            self.stage_kill += 1
                            if e.boss:
                                self.stage_kill_boss+=1
                        self.last_time_hit += reward_power
                        self.total_hit_power += reward_power
                        self.window_hit_power += reward_power

                        i.show = False
                        if e.health <= 0 and e.boss:
                            if self.visual and self.audio_enabled:
                                self.channel_crash.play(self.crash_sound)
                    if self.visual and self.audio_enabled and not self.channel_hit.get_busy():
                        self.channel_hit.play(self.hit_sound)
            if should_render:
                pl = self.count // 10
                if direction == 0:
                    self.window.blit(self.spritesheet[pl % 3], (self.player1.position_x - 25 - 1, self.player1.position_y - 25))
                elif direction == 2:
                    self.window.blit(self.spritesheet[pl % 3 + 3], (self.player1.position_x - 25 - 1, self.player1.position_y - 25))
                else:
                    self.window.blit(self.spritesheet[pl % 3 + 6], (self.player1.position_x - 25 - 1, self.player1.position_y - 25))

            if len(self.enemy_list) > 0:
                for i, e in enumerate(self.enemy_list):
                    e.update_bullets()
                    if e.boss:
                        for b in e.bullets:
                            b.record[2]*=0.9999
                    for b in e.bullets:
                        hit_radius = b.size + self.player1.size
                        dx = b.position_x - self.player1.position_x
                        dy = b.position_y - self.player1.position_y
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < hit_radius * hit_radius and self.player1.show:

                            self.player1.change_health(-b.power)
                            self.last_time_hurt += b.power
                            self.total_hurt_power += b.power
                            self.window_hurt_power += b.power

                            b.show = False
                        elif hit_radius * hit_radius < dist_sq < (hit_radius + 5) * (hit_radius + 5) and self.player1.show:
                            if self.visual and self.audio_enabled:
                                self.channel_close.play(self.close_sound)
                        if b.show and should_render:
                            pygame.draw.circle(self.window, b.color, (b.position_x, b.position_y), b.size)
                    if e.update(self.player1.position_x, self.player1.position_y) == False:
                        remove_en.append(i)
                for i in sorted(remove_en, reverse=True):
                    del self.enemy_list[i]
            self._handle_terminal_state(agent)
            if shift != 1 and should_render:
                pygame.draw.circle(self.window, self.player1.color, (self.player1.position_x, self.player1.position_y), self.player1.size)
            #self.draw_agent_zones(self.window, agent, (self.player1.position_x, self.player1.position_y))
            if should_render:
                self.show_heath(self.player1, True)
                self.draw_rl_input_overlay(agent)
            if self.visual:
                self.clock.tick(self.frame_cap)
            self.countfps+=1
            #print(agent.bullet_count)
            if self.countfps % 100 == 0:
                #print(f'FPS: {self.countfps / (time.time() - self.curtime)}:.2f, Qtable is not new: {agent.q_table!={}}')
                self.countfps = 0
                self.curtime= time.time()
            if self.count % self.log_every == 0:
                self._print_text_status(agent)
            if should_render:
                self._present()

    def move_randomly(self, x, y, speed):
        limitx = 590
        limity = 290
        min_boundary = 10  # 最小边界

        # 使用 self.rand 计算随机角度 θ ∈ [0, 2π)
        random.seed(self.rand)
        theta = random.uniform(0, 2 * math.pi)  # 随机角度
        dx = speed * math.cos(theta)  # x方向变化
        dy = speed * math.sin(theta)  # y方向变化

        # 计算新位置
        new_x = x + dx
        new_y = y + dy

        # 检查是否撞墙
        hit_wall = False
        if new_x < min_boundary or new_x > limitx:
            dx = -dx  # 水平反弹
            new_x = x + dx
            hit_wall = True
        if new_y < min_boundary or new_y > limity:
            dy = -dy  # 垂直反弹
            new_y = y + dy
            hit_wall = True

        # 如果撞墙，更新 self.rand
        if hit_wall:
            self.rand = random.randint(0, 2 ** 32 - 1)  # 生成新的随机种子

        return new_x, new_y

    def draw_agent_zones(self, surface, agent, player_pos, color_direction=(0, 255, 0), color_box=(255, 0, 0), width=1):
        """
        所有九宫格中心点距离为d，均为边长2*d正方形，中心点用圆点标记。
        """
        import pygame
        import math

        x, y = player_pos
        d = agent.direction_range
        diag = d / math.sqrt(2)
        directions = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y)
        ]

        for cx, cy in directions:
            pygame.draw.circle(surface, color_direction, (int(cx), int(cy)), 3, 0)
            rect = pygame.Rect(int(cx - d), int(cy - d), 2 * d, 2 * d)
            pygame.draw.rect(surface, color_box, rect, width)

        pygame.draw.circle(surface, (0, 0, 255), (int(x), int(y)), 4, 0)

    def draw_rl_input_overlay(self, agent):
        if self.overlay_mode == 0:
            return
        state = self._last_processed_state
        self._draw_rl_input_ranges(agent, state)
        if self.overlay_mode == 2:
            self._draw_rl_hud(agent, state)

    def _draw_rl_input_ranges(self, agent, state):
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.SRCALPHA)
        x = self.player1.position_x
        y = self.player1.position_y
        d = agent.direction_range
        diag = d / math.sqrt(2)
        directions = [
            (x - diag, y - diag), (x, y - d), (x + diag, y - diag),
            (x + d, y), (x + diag, y + diag), (x, y + d),
            (x - diag, y + diag), (x - d, y), (x, y),
        ]
        action_delta = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0), (0, 0),
        ]
        decision = getattr(agent, "last_decision", {})
        action = decision.get("action")

        inner_rect = pygame.Rect(
            agent.wall_repulse_dist,
            agent.wall_repulse_dist,
            self.WINDOW_WIDTH - agent.wall_repulse_dist * 2,
            self.WINDOW_HEIGHT - agent.wall_repulse_dist * 2,
        )
        pygame.draw.rect(overlay, (255, 120, 0, 90), inner_rect, 1)

        shot_angle = math.radians(agent.shot_cone_angle_degrees)
        left_end = (x - math.sin(shot_angle) * agent.shot_range, y - math.cos(shot_angle) * agent.shot_range)
        right_end = (x + math.sin(shot_angle) * agent.shot_range, y - math.cos(shot_angle) * agent.shot_range)
        center_end = (x, y - agent.shot_range)
        pygame.draw.polygon(overlay, (255, 220, 80, 24), [(x, y), left_end, right_end])
        pygame.draw.line(overlay, (255, 220, 80, 160), (x, y), left_end, 2)
        pygame.draw.line(overlay, (255, 220, 80, 160), (x, y), right_end, 2)
        pygame.draw.line(overlay, (255, 245, 180, 140), (x, y), center_end, 1)

        for idx, (dx, dy) in enumerate(action_delta):
            if dx == 0 and dy == 0:
                continue
            norm = math.hypot(dx, dy)
            end_x = x + dx / norm * agent.max_lookahead
            end_y = y + dy / norm * agent.max_lookahead
            pygame.draw.line(overlay, (80, 180, 255, 60), (x, y), (end_x, end_y), 1)

        for idx, (cx, cy) in enumerate(directions):
            rect = pygame.Rect(int(cx - d), int(cy - d), int(2 * d), int(2 * d))
            color = (0, 255, 120, 210) if idx == action else (255, 255, 255, 130)
            pygame.draw.rect(overlay, color, rect, 1)
            count = int(state[idx]) if state else 0
            label = self.debug_font_small.render(str(count), True, color[:3])
            overlay.blit(label, (int(cx + 3), int(cy - 10)))

        pygame.draw.circle(overlay, (0, 140, 255, 220), (int(x), int(y)), 4)
        self.window.blit(overlay, (0, 0))

    def _draw_rl_hud(self, agent, state):
        decision = getattr(agent, "last_decision", {})
        candidates = decision.get("candidate_actions", tuple(range(agent.action_size)))
        q_values = decision.get("q_values", tuple(0.0 for _ in range(agent.action_size)))
        adjusted_values = decision.get("adjusted_values", q_values)
        action_priors = decision.get("action_priors", tuple(0.0 for _ in range(agent.action_size)))
        action = decision.get("action", self.action)
        action_label = decision.get("action_label", str(action))
        bullet_counts = tuple(int(v) for v in state[:9]) if state else tuple(0 for _ in range(9))
        threat_scores = tuple(float(v) for v in state[9:18]) if state else tuple(0.0 for _ in range(9))
        aim_features = tuple(state[24:29]) if state and len(state) >= 29 else (0, 0, 0, 0.0, 1.0)
        reward = getattr(agent, "last_reward_components", {})

        lines = [
            f"RL policy: {decision.get('policy_mode', 'pure-q')} | hard rules: {'ON' if agent.use_hard_rules else 'OFF'}",
            f"Candidates: {list(candidates)} | action: {action} {action_label} | eps: {agent.epsilon:.3f}",
            f"Input range: 9 boxes {agent.direction_range * 2}px, threat lookahead {agent.max_lookahead}px, wall band {agent.wall_repulse_dist}px",
            f"Shot cone: +/-{agent.shot_cone_angle_degrees}deg, range {agent.shot_range}px | L/C/R enemies: {int(aim_features[0])}/{int(aim_features[1])}/{int(aim_features[2])}",
            f"Aim align: {float(aim_features[3]):.2f} | nearest in cone: {float(aim_features[4]):.2f} range",
            f"Input dims: bullets[0:9], threat[9:18], player x/y, enemy rel x/y, wall w/h, aim[24:28]",
            f"Bullets: {bullet_counts}",
            "Threat: " + ", ".join(f"{v:.1f}" if v < 999999 else "inf" for v in threat_scores),
            "Q: " + ", ".join(f"{i}:{q_values[i]:.1f}" for i in range(agent.action_size)),
            "Prior: " + ", ".join(f"{i}:{action_priors[i]:.1f}" for i in range(agent.action_size)),
            "Q+prior: " + ", ".join(f"{i}:{adjusted_values[i]:.1f}" for i in range(agent.action_size)),
            "Reward: "
            + f"hit {reward.get('hit', 0.0):.1f}, hurt {reward.get('hurt', 0.0):.1f}, "
            + f"aim {reward.get('aim', 0.0):.1f}, wall {reward.get('wall', 0.0):.1f}",
            f"UI scale: {self.ui_scale:.2f} | logic size: {self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}",
        ]

        padding = 6
        line_height = 18
        width = 560
        height = padding * 2 + line_height * len(lines)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        for i, line in enumerate(lines):
            text = self.debug_font.render(line, True, (230, 235, 240))
            panel.blit(text, (padding, padding + i * line_height))
        self.window.blit(panel, (8, 8))
# 用于直接运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without a visible pygame window.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cpu, mps, cuda:0, ...")
    parser.add_argument("--checkpoint", default="current_best_dqn.pth")
    parser.add_argument("--no-load", action="store_true", help="Start a new model instead of loading checkpoint.")
    parser.add_argument("--render-every", type=int, default=5, help="Visible mode only: draw every N logic frames.")
    parser.add_argument("--log-every", type=int, default=1000, help="Print training status every N logic frames.")
    parser.add_argument("--log-file", default="training_log.jsonl", help="Append JSONL training metrics here. Empty disables file logging.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size for each gradient update.")
    parser.add_argument("--train-every", type=int, default=4, help="Run gradient updates every N agent steps.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient updates to run at each training point.")
    args = parser.parse_args()

    agent = STGAgent(
        device=args.device,
        batch_size=args.batch_size,
        train_every=args.train_every,
        gradient_steps=args.gradient_steps,
    )

    model_loaded = False
    if not args.no_load:
        try:
            agent.load(args.checkpoint)
            model_loaded = True
            print(f"Loaded {args.checkpoint} on {agent.device}.", flush=True)
        except FileNotFoundError:
            print(f"No saved model found at {args.checkpoint}, starting fresh on {agent.device}.", flush=True)
        except Exception as exc:
            print(f"Could not load {args.checkpoint}: {exc}. Starting fresh on {agent.device}.", flush=True)
    else:
        print(f"Starting fresh on {agent.device}.", flush=True)

    game = Train(
        headless=args.headless,
        render_every=args.render_every,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        log_file=args.log_file or None,
        load_training_state=model_loaded,
    )
    game.run()
