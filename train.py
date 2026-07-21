import math
import argparse
import csv
import json
import os
import random
import sys
import time

if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch
from sympy.physics.units import action

import character
import img
from drl_agent import ACTION_LABELS
from drl_agent import DRLAgent as STGAgent


REWARD_SCHEMA_VERSION = 4


class Train:
    def __init__(
            self,
            headless=False,
            render_every=5,
            log_every=1000,
            checkpoint="current_best_dqn.pth",
            log_file="training_log.csv",
            load_training_state=True,
            training_enabled=None,
            random_first_action=None):
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
        self.best_checkpoint = self._best_checkpoint_path(checkpoint)
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        self.pygame = pygame
        self.pygame.init()
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 900
        self.visual = not self.headless
        self.training_enabled = self.headless if training_enabled is None else training_enabled
        self.randomize_first_action = self.training_enabled if random_first_action is None else random_first_action
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
        self.fast_training = self.headless
        self.frame_cap = 0 if self.headless else 60
        self.render_every = render_every
        self.log_every = log_every
        self.audio_enabled = self.visual
        self._image_cache = {}
        self.player1 = character.player(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2,
                                        10, 5, True,
                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT - 5], 100, (20, 255, 255))
        self.clock = pygame.time.Clock()
        self.enemy_list = []
        self.curtime= time.time()
        self.count = 0
        self.log_step_offset = 0
        self.enemy_type = 0
        self.learncount=0
        self.learntimes=0
        self.stage=0
        self.stage_kill=0
        self.stage_kill_boss=0
        self.best_stage = 0
        self.best_stage_kill = 0
        self.best_stage_kill_boss = 0
        self.win_count = 0
        self.best_avg_reward = float("-inf")
        self.best_model_avg_reward = float("-inf")
        self.best_model_score = None
        self.best_rollback_bad_logs = 0
        self.best_rollback_count = 0
        self.best_rollback_cooldown_until = 0
        self.best_rollback_margin = 60.0
        self.best_rollback_patience = 3
        self.best_rollback_min_steps = 30000
        self.episode = 0
        self.death_resets = 0
        self.boss_stall_resets = 0
        self.last_terminal_reason = ""
        self.last_time_boss_damage = 0
        self.last_time_boss_reward_damage = 0
        self.random_first_action_pending = self.randomize_first_action
        self.window_boss_damage = 0.0
        self.window_boss_reward_damage = 0.0
        self.window_min_boss_hp_pct = None
        self.last_seen_boss_hp_pct = 0.0
        self.boss_no_damage_frames = 0
        self.boss_no_hit_frames = 0
        self.boss_low_hp_no_hit_frames = 0
        self.boss_stall_hp_pct = 0.35
        self.boss_stall_no_hit_limit = 2400
        self.boss_stall_low_hp_no_hit_limit = 1200
        self.hit_death_extra_penalty = -5000.0
        self.low_boss_survival_hp_pct = 0.60
        self.low_boss_survival_reward_scale = 18.0
        self.final_boss_survival_hp_pct = 0.30
        self.final_boss_survival_reward_scale = 25.0
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
        self.log_header_written = bool(log_file and os.path.exists(log_file) and os.path.getsize(log_file) > 0)
        self.log_file = open(log_file, "a", buffering=1, encoding="utf-8", newline="") if log_file else None
        self.log_writer = None
        self.log_fieldnames = None
        self.training_state_path = f"{checkpoint}.train_state.json" if checkpoint else None
        if load_training_state:
            self._load_training_state()

        # 璧勬簮鍔犺浇
        self.spritesheet = img.load_character_spritesheet("src/img_1.png", 4, 3, 50, 50) if self.visual else []
        self.volume = .5 if self.audio_enabled else 0.0
        if self.audio_enabled:
            self.close_sound = character.safe_sound("src/涓滄柟鍘熶綔闊虫晥/缁€闀挎摝寮?wav", self.volume)
            self.hit_sound = character.safe_sound("src/涓滄柟鍘熶綔闊虫晥/鑾庤帋鐏寮瑰懡涓?wav", self.volume)
            self.crash_sound = character.safe_sound("src/涓滄柟鍘熶綔闊虫晥/鍑荤牬boss.wav", self.volume)
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

        # 鍙皟鍙傛暟 for enemytype4
        self.enemytype4_cfg = {
            "enemy0_num": 2,
            "enemy1_num": 2,
            "boss0_num": 1,
            "enemy0_freq": 200,
            "enemy1_freq": 300,
            "boss0_unique": True, # 鍙厑璁稿悓鏃跺瓨鍦ㄤ竴涓猙oss
        }

    @staticmethod
    def _best_checkpoint_path(checkpoint):
        if not checkpoint:
            return None
        root, ext = os.path.splitext(checkpoint)
        if not ext:
            ext = ".pth"
        return f"{root}.best{ext}"

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
            # 4绫诲瀷涓?0/1/3 鐨勭粨鍚堬紝涓嶆柊寤虹被锛岃€屾槸璋冪敤鍚勮嚜鐢熸垚
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
        elif event.key == pygame.K_m:
            self._toggle_audio()

    def _toggle_audio(self):
        if not self.visual:
            return
        self.audio_enabled = not self.audio_enabled
        self.volume = 0.5 if self.audio_enabled else 0.0
        for sound in (self.sound, self.hit_sound, self.close_sound, self.crash_sound):
            if sound is not None:
                sound.set_volume(self.volume)
        if self.audio_enabled:
            self.channel_sound.play(self.sound, -1)
        else:
            self.channel_sound.stop()

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
        window_steps = max(1, self.count - self.last_log_count)
        sps = window_steps / window_elapsed
        accuracy = self.total_hit_power / max(1.0, self.total_shot_power)
        window_accuracy = self.window_hit_power / max(1.0, self.window_shot_power)
        boss_pct = self.window_min_boss_hp_pct
        if boss_pct is None:
            boss_pct = self._boss_hp_pct()
        decision = getattr(agent, "last_decision", {})
        loss = decision.get("loss")
        avg_reward = self.window_reward / window_steps
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
            "step": self.log_step_offset + self.count,
            "session_step": self.count,
            "wins": self.win_count,
            "stage": self.stage,
            "stage_kill": self.stage_kill,
            "stage_boss_kill": self.stage_kill_boss,
            "boss_damage": self.window_boss_reward_damage,
            "boss_real_damage": self.window_boss_damage,
            "avg_reward": avg_reward,
            "best_avg_reward": self.best_avg_reward,
            "reward": self.last_reward,
            "window_reward": self.window_reward,
            "death_resets": self.death_resets,
            "boss_stall_resets": self.boss_stall_resets,
            "last_terminal_reason": self.last_terminal_reason,
            "progress": progress,
            "best_progress": best_progress,
            "boss_hp_pct": boss_pct,
            "sps": sps,
            "epsilon": getattr(agent, "epsilon", 0.0),
            "loss": loss,
            "replay_size": decision.get("replay_size", 0),
            "updates": decision.get("updates", 0),
            "episode": self.episode,
            "elapsed_sec": elapsed,
            "device": str(decision.get("device", getattr(agent, "device", "unknown"))),
            "player_hp": self.player1.health,
            "player_full_hp": self.player1.full_health,
            "accuracy": accuracy,
            "window_accuracy": window_accuracy,
            "hit_power": self.total_hit_power,
            "hurt_power": self.total_hurt_power,
            "shot_power": self.total_shot_power,
            "window_hit_power": self.window_hit_power,
            "window_hurt_power": self.window_hurt_power,
            "window_shot_power": self.window_shot_power,
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
        if self.log_writer is None:
            self.log_fieldnames = list(metrics.keys())
            self.log_writer = csv.DictWriter(
                self.log_file,
                fieldnames=self.log_fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            if not self.log_header_written:
                self.log_writer.writeheader()
                self.log_header_written = True
        self.log_writer.writerow(metrics)
        self.log_file.flush()

    def _save_training_state(self):
        if not self.training_state_path:
            return
        state = {
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "total_reward": self.total_reward,
            "reward_steps": self.reward_steps,
            "log_step": self.log_step_offset + self.count,
            "best_stage": self.best_stage,
            "best_stage_kill": self.best_stage_kill,
            "best_stage_kill_boss": self.best_stage_kill_boss,
            "stage": self.stage,
            "stage_kill": self.stage_kill,
            "stage_kill_boss": self.stage_kill_boss,
            "enemy_type": self.enemy_type,
            "win_count": self.win_count,
            "best_avg_reward": self.best_avg_reward,
            "best_model_avg_reward": self.best_model_avg_reward,
            "best_model_score": self.best_model_score,
            "best_rollback_bad_logs": self.best_rollback_bad_logs,
            "best_rollback_count": self.best_rollback_count,
            "best_rollback_cooldown_until": self.best_rollback_cooldown_until,
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
        self.log_step_offset = int(state.get("log_step", state.get("reward_steps", self.log_step_offset)))
        self.best_stage = int(state.get("best_stage", self.best_stage))
        self.best_stage_kill = int(state.get("best_stage_kill", self.best_stage_kill))
        self.best_stage_kill_boss = int(state.get("best_stage_kill_boss", self.best_stage_kill_boss))
        self.stage = int(state.get("stage", state.get("best_stage", self.stage)))
        self.stage_kill = int(state.get("stage_kill", self.stage_kill))
        self.stage_kill_boss = int(state.get("stage_kill_boss", self.stage_kill_boss))
        self.enemy_type = int(state.get("enemy_type", self._enemy_type_for_stage(self.stage)))
        self.win_count = int(state.get("win_count", self.win_count))
        if int(state.get("reward_schema_version", -1)) == REWARD_SCHEMA_VERSION:
            self.best_avg_reward = float(state.get("best_avg_reward", self.best_avg_reward))
            self.best_model_avg_reward = float(state.get("best_model_avg_reward", self.best_model_avg_reward))
            saved_score = state.get("best_model_score")
            if isinstance(saved_score, list):
                self.best_model_score = tuple(saved_score)
            self.best_rollback_bad_logs = int(state.get("best_rollback_bad_logs", self.best_rollback_bad_logs))
            self.best_rollback_count = int(state.get("best_rollback_count", self.best_rollback_count))
            self.best_rollback_cooldown_until = int(state.get("best_rollback_cooldown_until", self.best_rollback_cooldown_until))
        self.episode = int(state.get("episode", self.episode))
        self.death_resets = int(state.get("death_resets", self.death_resets))
        self.boss_stall_resets = int(state.get("boss_stall_resets", self.boss_stall_resets))

    def _reset_log_window(self):
        self.window_reward = 0.0
        self.window_hit_power = 0.0
        self.window_hurt_power = 0.0
        self.window_shot_power = 0.0
        self.window_boss_damage = 0.0
        self.window_boss_reward_damage = 0.0
        self.window_min_boss_hp_pct = None

    def _update_boss_time_penalty(self):
        boss_alive = any(
            getattr(enemy, "boss", False) and getattr(enemy, "show", True)
            for enemy in self.enemy_list
        )
        if not boss_alive:
            self.boss_alive_frames = 0
            self.boss_no_damage_frames = 0
            self.last_boss_time_penalty = 0.0
            return 0.0

        self.boss_alive_frames += 1
        if self.last_time_boss_reward_damage > 0:
            self.boss_no_damage_frames = 0
            self.last_boss_time_penalty = 0.0
            return 0.0
        self.boss_no_damage_frames += 1
        penalty = -min(6.0, 0.01 * self.boss_no_damage_frames)
        self.last_boss_time_penalty = penalty
        self.total_boss_time_penalty += penalty
        return penalty

    def _boss_hp_sample(self):
        boss_hp = 0.0
        boss_max_hp = 0.0
        for enemy in self.enemy_list:
            if getattr(enemy, "boss", False) and getattr(enemy, "show", True):
                boss_hp += max(0.0, getattr(enemy, "health", 0.0))
                boss_max_hp += max(1.0, getattr(enemy, "full_health", 1.0))
        if not boss_max_hp:
            return None
        return boss_hp / boss_max_hp

    def _boss_hp_pct(self):
        sample = self._boss_hp_sample()
        return sample if sample is not None else 0.0

    def _observe_boss_hp_window(self):
        sample = self._boss_hp_sample()
        if sample is None:
            return
        self.last_seen_boss_hp_pct = sample
        if self.window_min_boss_hp_pct is None:
            self.window_min_boss_hp_pct = sample
        else:
            self.window_min_boss_hp_pct = min(self.window_min_boss_hp_pct, sample)

    def _non_boss_hit_reward(self):
        return max(0.0, self.last_time_hit - self.last_time_boss_reward_damage)


    def _death_penalty(self):
        boss_pct = self._boss_hp_pct()
        return -9000.0 - 7000.0 * boss_pct

    def _boss_attack_reward(self):
        if self.last_time_boss_reward_damage <= 0:
            return 0.0
        boss_pct = self._boss_hp_pct()
        progress_multiplier = 0.25 + 1.75 * (1.0 - boss_pct)
        scale = 35.0 if self.stage == 4 else 20.0
        return self.last_time_boss_reward_damage * scale * progress_multiplier

    def _low_boss_survival_reward(self):
        if self.stage != 4:
            return 0.0
        if self.last_time_hurt > 0 or self.last_time_boss_reward_damage <= 0:
            return 0.0
        boss_pct = self._boss_hp_pct()
        if boss_pct <= 0.0 or boss_pct >= self.low_boss_survival_hp_pct:
            return 0.0
        low_hp_factor = (self.low_boss_survival_hp_pct - boss_pct) / self.low_boss_survival_hp_pct
        reward = self.last_time_boss_reward_damage * self.low_boss_survival_reward_scale * low_hp_factor
        if boss_pct < self.final_boss_survival_hp_pct:
            final_factor = (self.final_boss_survival_hp_pct - boss_pct) / self.final_boss_survival_hp_pct
            reward += self.last_time_boss_reward_damage * self.final_boss_survival_reward_scale * final_factor
        return reward

    def _update_best_checkpoint_guard(self, agent, metrics):
        if not self.training_enabled or not self.best_checkpoint:
            return
        score = self._checkpoint_score(metrics)
        avg_reward = float(metrics["avg_reward"])
        if self.best_model_score is None or score > self.best_model_score:
            self.best_model_score = score
            self.best_model_avg_reward = avg_reward
            self.best_rollback_bad_logs = 0
            agent.save(self.best_checkpoint)

    @staticmethod
    def _checkpoint_score(metrics):
        stage = int(metrics.get("stage", 0) or 0)
        boss_pct = float(metrics.get("boss_hp_pct", 0.0) or 0.0)
        boss_progress = max(0.0, 1.0 - boss_pct) if stage == 4 and boss_pct > 0.0 else 0.0
        return (
            int(metrics.get("wins", 0) or 0),
            stage,
            int(metrics.get("stage_boss_kill", 0) or 0),
            boss_progress,
            float(metrics.get("boss_damage", 0.0) or 0.0),
            float(metrics.get("boss_real_damage", 0.0) or 0.0),
            float(metrics.get("avg_reward", 0.0) or 0.0),
        )

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
        if self.last_time_boss_damage > 0 or self.last_time_hit > 0:
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
        elif reason == "hit_death":
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
        self.boss_no_damage_frames = 0
        self.last_boss_time_penalty = 0.0
        self.boss_no_hit_frames = 0
        self.boss_low_hp_no_hit_frames = 0
        self._had_enemies_for_reward = False
        self._last_processed_state = None
        self._last_action = None
        self.action = 8
        self.random_first_action_pending = self.randomize_first_action
        self.last_time_hit = 0
        self.last_time_hurt = 0
        self.last_time_boss_damage = 0
        self.last_time_boss_reward_damage = 0

    def _learn_terminal_transition(self, agent, reason, penalty):
        if not self.training_enabled:
            return
        if self._last_processed_state is None or self._last_action is None:
            return
        terminal_state, reward = agent._process_game_state(
            self.enemy_list,
            [self.player1.position_x, self.player1.position_y],
            self._non_boss_hit_reward(),
            self.last_time_hurt,
            self.WINDOW_WIDTH,
            self.WINDOW_HEIGHT,
        )
        boss_attack_reward = self._boss_attack_reward()
        low_boss_survival_reward = self._low_boss_survival_reward()
        reward += boss_attack_reward
        reward += low_boss_survival_reward
        reward += penalty
        components = getattr(agent, "last_reward_components", None)
        if isinstance(components, dict):
            components["boss_attack"] = boss_attack_reward
            components["low_boss_survival"] = low_boss_survival_reward
            components["terminal"] = penalty
            components["total"] = reward
        self.last_reward = reward
        self.total_reward += reward
        self.reward_steps += 1
        self.window_reward += reward
        agent.learn(reward, terminal_state, self._last_action, done=True)
        agent.last_state = None
        agent.last_action = None

    def _handle_terminal_state(self, agent):
        if self.last_time_hurt > 0:
            self._learn_terminal_transition(agent, "hit_death", self._death_penalty() + self.hit_death_extra_penalty)
            self._reset_episode("hit_death")
            return True
        if self.player1.health <= 0:
            self._learn_terminal_transition(agent, "death", self._death_penalty())
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
                f"win={metrics['wins']}",
                f"avg={metrics['avg_reward']:.1f}",
                f"best_avg={metrics['best_avg_reward']:.1f}",
                f"reset={metrics['death_resets']}/{metrics['boss_stall_resets']}",
            ]
        )
        padding = max(0, self._last_status_width - len(status))
        print("\r" + status + (" " * padding), end="", flush=True)
        self._last_status_width = len(status)
        self._write_log_record(metrics)
        self._update_best_checkpoint_guard(agent, metrics)
        if self.training_enabled:
            self._save_training_state()
        self._reset_log_window()
        self.last_log_count = self.count
        self.last_log_time = time.time()

    def action1(self):
        self.learncount+=1
        self.learncount%= 10000
        if self.learntimes%1==0:
            if self.training_enabled and self.learncount==1:
                #self.learntimes+=1
                agent.save(self.checkpoint)  # 淇濆瓨妯″瀷鐘舵€?                self._save_training_state()
            # 1. 濡傛灉瀛樺湪涓婁竴娆＄殑鐘舵€佸拰鍔ㄤ綔锛屽厛杩涜瀛︿範
            current_state = None
            if self.training_enabled and self._last_processed_state is not None and self._last_action is not None:
                # 鑾峰彇褰撳墠鐘舵€?
                current_state, reward = agent._process_game_state(
                    self.enemy_list,
                    [self.player1.position_x, self.player1.position_y],
                    self._non_boss_hit_reward(),
                    self.last_time_hurt,
                    self.WINDOW_WIDTH,
                    self.WINDOW_HEIGHT
                )
                enemies_alive_for_reward = len(self.enemy_list) > 0
                if self._had_enemies_for_reward and not enemies_alive_for_reward:
                    reward += 100  # 娑堢伃鎵€鏈夋晫浜哄鍔?
                self._had_enemies_for_reward = enemies_alive_for_reward
                boss_attack_reward = self._boss_attack_reward()
                low_boss_survival_reward = self._low_boss_survival_reward()
                reward += boss_attack_reward
                reward += low_boss_survival_reward
                reward += self._update_boss_time_penalty()
                components = getattr(agent, "last_reward_components", None)
                if isinstance(components, dict):
                    components["boss_attack"] = boss_attack_reward
                    components["low_boss_survival"] = low_boss_survival_reward
                    components["boss_time"] = self.last_boss_time_penalty
                    components["total"] = reward
                self.last_reward = reward
                self.total_reward += reward
                self.reward_steps += 1
                self.window_reward += reward

                # 杩涜Q-learning鏇存柊
                agent.learn(reward, current_state, self._last_action)

            # 2. 澶勭悊褰撳墠鐘舵€佸苟鑾峰彇鏂板姩浣?
            if current_state is None:
                processed_state, current_reward = agent._process_game_state(
                    self.enemy_list,
                    [self.player1.position_x, self.player1.position_y],
                    self._non_boss_hit_reward(),
                    self.last_time_hurt,
                    self.WINDOW_WIDTH,
                    self.WINDOW_HEIGHT
                )
            else:
                processed_state = current_state
            #print(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

            # 3. 鑾峰彇鍔ㄤ綔
            action = agent.get_action(processed_state,self.enemy_list)
            if self.random_first_action_pending:
                action = random.randrange(agent.action_size)
                if isinstance(getattr(agent, "last_decision", None), dict):
                    agent.last_decision["action"] = action
                    agent.last_decision["action_label"] = ACTION_LABELS[action]
                    agent.last_decision["explored"] = True
                self.random_first_action_pending = False

            # 4. 瀛樺偍褰撳墠鐘舵€佸拰鍔ㄤ綔
            self._last_processed_state = processed_state
            self._last_action = action
            self.action = action
            # if self.player1.level<4:
            #     self.player1.level+=(self.last_time_hit*0.001)*(4-self.player1.level)
            # if self.player1.level>0:
            #     self.player1.level-= self.last_time_hurt*(4-self.player1.level)*0.5
            # 5. 閲嶇疆鍛戒腑/鍙椾激璁℃暟鍣?
            self.last_time_hit = 0
            self.last_time_hurt = 0
            self.last_time_boss_damage = 0
            self.last_time_boss_reward_damage = 0
            #print(action)

    def run(self):
        if self.visual and self.audio_enabled:
            self.channel_sound.play(self.sound, -1)
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
            if self.action == 0:  # 宸︿笂
                self.player1.move_x(-shift)
                self.player1.move_y(-shift)
            elif self.action == 1:  # 涓?
                self.player1.move_y(-shift)
            elif self.action == 2:  # 鍙充笂
                self.player1.move_x(shift)
                self.player1.move_y(-shift)
            elif self.action == 3:  # 鍙?
                self.player1.move_x(shift)
            elif self.action == 4:  # 鍙充笅
                self.player1.move_x(shift)
                self.player1.move_y(shift)
            elif self.action == 5:  # 涓?
                self.player1.move_y(shift)
            elif self.action == 6:  # 宸︿笅
                self.player1.move_x(-shift)
                self.player1.move_y(shift)
            elif self.action == 7:  # 宸?
                self.player1.move_x(-shift)
            elif self.action == 8:  # 涓嶅姩
                pass


            # AI/璁粌鎺ュ彛

            self.last_time_hit = 0
            self.last_time_hurt = 0
            self.last_time_boss_damage = 0
            self.last_time_boss_reward_damage = 0

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
            #hit妫€娴?
            for i1, i in enumerate(self.player1.bullets):
                if should_render:
                    pygame.draw.circle(self.window, i.color, (i.position_x, i.position_y), i.size)
                for e in self.enemy_list:
                    hit_radius = e.size + i.size
                    dx = i.position_x - e.position_x
                    dy = e.position_y - i.position_y
                    if dx * dx + dy * dy < hit_radius * hit_radius and e.show:
                        was_alive = e.health > 0
                        e.change_health(-i.power)
                        reward_power = getattr(i, "reward_power", i.power)
                        if e.boss:
                            self.last_time_boss_damage += i.power
                            self.window_boss_damage += i.power
                            self.last_time_boss_reward_damage += reward_power
                            self.window_boss_reward_damage += reward_power
                        if was_alive and e.health <= 0:
                            self.stage_kill += 1
                            if e.boss:
                                self.stage_kill_boss += 1
                                if self.stage == 4:
                                    self.win_count += 1
                        self.last_time_hit += reward_power
                        self.total_hit_power += reward_power
                        self.window_hit_power += reward_power

                        i.show = False
                        if e.health <= 0 and e.boss:
                            if self.visual and self.audio_enabled:
                                self.channel_crash.play(self.crash_sound)
                    if self.visual and self.audio_enabled and not self.channel_hit.get_busy():
                        self.channel_hit.play(self.hit_sound)
            self._observe_boss_hp_window()
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
                            self.player1.health = 0
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
        min_boundary = 10  # 鏈€灏忚竟鐣?

        # 浣跨敤 self.rand 璁＄畻闅忔満瑙掑害 胃 鈭?[0, 2蟺)
        random.seed(self.rand)
        theta = random.uniform(0, 2 * math.pi)  # 闅忔満瑙掑害
        dx = speed * math.cos(theta)  # x鏂瑰悜鍙樺寲
        dy = speed * math.sin(theta)  # y鏂瑰悜鍙樺寲

        # 璁＄畻鏂颁綅缃?
        new_x = x + dx
        new_y = y + dy

        # 妫€鏌ユ槸鍚︽挒澧?
        hit_wall = False
        if new_x < min_boundary or new_x > limitx:
            dx = -dx  # 姘村钩鍙嶅脊
            new_x = x + dx
            hit_wall = True
        if new_y < min_boundary or new_y > limity:
            dy = -dy  # 鍨傜洿鍙嶅脊
            new_y = y + dy
            hit_wall = True

        # 濡傛灉鎾炲锛屾洿鏂?self.rand
        if hit_wall:
            self.rand = random.randint(0, 2 ** 32 - 1)  # 鐢熸垚鏂扮殑闅忔満绉嶅瓙

        return new_x, new_y

    def draw_agent_zones(self, surface, agent, player_pos, color_direction=(0, 255, 0), color_box=(255, 0, 0), width=1):
        """
        鎵€鏈変節瀹牸涓績鐐硅窛绂讳负d锛屽潎涓鸿竟闀?*d姝ｆ柟褰紝涓績鐐圭敤鍦嗙偣鏍囪銆?
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

        local_radius = int(agent.local_world_radius)
        local_rect = pygame.Rect(
            int(x - local_radius),
            int(y - local_radius),
            local_radius * 2,
            local_radius * 2,
        ).clip(pygame.Rect(0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.draw.rect(overlay, (80, 220, 255, 42), local_rect, 0)
        pygame.draw.rect(overlay, (80, 220, 255, 210), local_rect, 2)
        for ratio, alpha in ((0.25, 150), (0.5, 110), (0.75, 80), (1.0, 60)):
            r = int(local_radius * ratio)
            pygame.draw.circle(overlay, (80, 220, 255, alpha), (int(x), int(y)), r, 1)
        pygame.draw.line(overlay, (80, 220, 255, 120), (int(x - local_radius), int(y)), (int(x + local_radius), int(y)), 1)
        pygame.draw.line(overlay, (80, 220, 255, 120), (int(x), int(y - local_radius)), (int(x), int(y + local_radius)), 1)
        label = self.debug_font_small.render(
            f"local CNN lens {agent.local_grid_size}x{agent.local_grid_size}, radius {local_radius}px",
            True,
            (120, 235, 255),
        )
        overlay.blit(label, (max(4, int(local_rect.left)), max(4, int(local_rect.top) - 18)))

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
        aim_features = tuple(state[6:11]) if state and len(state) >= 11 else (0, 0, 0, 0.0, 1.0)
        reward = getattr(agent, "last_reward_components", {})

        lines = [
            f"RL policy: {decision.get('policy_mode', 'pure-q')} | hard rules: {'ON' if agent.use_hard_rules else 'OFF'}",
            f"Candidates: {list(candidates)} | action: {action} {action_label} | eps: {agent.epsilon:.3f}",
            f"Local CNN lens: {agent.local_grid_size}x{agent.local_grid_size}, radius {agent.local_world_radius}px | wall band {agent.wall_repulse_dist}px",
            f"Shot cone: +/-{agent.shot_cone_angle_degrees}deg, range {agent.shot_range}px | L/C/R enemies: {int(aim_features[0])}/{int(aim_features[1])}/{int(aim_features[2])}",
            f"Aim align: {float(aim_features[3]):.2f} | nearest in cone: {float(aim_features[4]):.2f} range",
            "Input dims: player x/y, enemy rel x/y, enemy/bullet counts, hit/hurt, aim, wall distance",
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
# 鐢ㄤ簬鐩存帴杩愯
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without a visible pygame window.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cpu, mps, cuda:0, ...")
    parser.add_argument("--checkpoint", default="current_best_dqn.pth")
    parser.add_argument("--no-load", action="store_true", help="Start a new model instead of loading checkpoint.")
    parser.add_argument("--render-every", type=int, default=5, help="Visible mode only: draw every N logic frames.")
    parser.add_argument("--log-every", type=int, default=1000, help="Print training status every N logic frames.")
    parser.add_argument("--log-file", default="training_log.csv", help="Append CSV training metrics here. Empty disables file logging.")
    parser.add_argument("--batch-size", type=int, default=None, help="Replay batch size for each gradient update.")
    parser.add_argument("--train-every", type=int, default=None, help="Run gradient updates every N agent steps.")
    parser.add_argument("--gradient-steps", type=int, default=None, help="Gradient updates to run at each training point.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Torch CPU threads. Default: balanced in headless, 1 in display mode.")
    parser.add_argument("--eval-epsilon", type=float, default=None, help="Display mode exploration rate. Default keeps checkpoint epsilon.")
    parser.add_argument("--greedy", action="store_true", help="Display mode only: force epsilon=0 and disable first-action randomization.")
    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size is not None else (128 if args.headless else 64)
    train_every = args.train_every if args.train_every is not None else 4
    gradient_steps = args.gradient_steps if args.gradient_steps is not None else 1
    torch_threads = args.torch_threads
    if torch_threads is None:
        torch_threads = max(1, min((os.cpu_count() or 2) - 1, 8)) if args.headless else 1
    torch.set_num_threads(torch_threads)

    agent = STGAgent(
        device=args.device,
        batch_size=batch_size,
        train_every=train_every,
        gradient_steps=gradient_steps,
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

    if not args.headless:
        if args.greedy:
            agent.epsilon = 0.0
        elif args.eval_epsilon is not None:
            agent.epsilon = max(0.0, min(1.0, args.eval_epsilon))
        print(
            f"Display mode: training disabled, epsilon={agent.epsilon:.3f}. "
            "Use --greedy for deterministic inference. Press M to toggle audio.",
            flush=True,
        )
    else:
        print(
            f"Training mode: batch={batch_size}, train_every={train_every}, "
            f"gradient_steps={gradient_steps}, torch_threads={torch_threads}.",
            flush=True,
        )

    game = Train(
        headless=args.headless,
        render_every=args.render_every,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        log_file=args.log_file or None,
        load_training_state=not args.no_load,
        training_enabled=args.headless,
        random_first_action=args.headless or (not args.greedy and agent.epsilon > 0.0),
    )
    game.run()
