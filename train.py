import math
import random
import sys
import time
import pygame

import character
import img
from RLmodel import STGAgent


class Train:
    def __init__(self):
        self.last_time_hit=0
        self.countfps=0
        self.last_time_hurt = 0
        self._last_processed_state= None
        self._last_action = None
        self.action=8
        self.pygame = pygame
        self.pygame.init()
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 900
        self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.pygame.display.set_caption("ThreeBrilliance")
        self.player1 = character.player(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2,
                                        10, 5, True,
                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT - 5], 100, (20, 255, 255))
        self.clock = pygame.time.Clock()
        self.enemy_list = []
        self.curtime= time.time()
        self.count = 0
        self.enemy_type = 3
        self.learncount=0
        self.learntimes=0

        # 资源加载
        self.spritesheet = img.load_character_spritesheet("src/img_1.png", 4, 3, 50, 50)
        self.close_sound = pygame.mixer.Sound("src/东方原作音效/绀长擦弹.wav")
        self.hit_sound = pygame.mixer.Sound("src/东方原作音效/莎莎火箭弹命中.wav")
        self.crash_sound = pygame.mixer.Sound("src/东方原作音效/击破boss.wav")
        self.sound = pygame.mixer.Sound("src/th15_13.mp3")
        self.channel_sound = pygame.mixer.Channel(0)
        self.channel_hit = pygame.mixer.Channel(1)
        self.channel_close = pygame.mixer.Channel(2)
        self.channel_crash = pygame.mixer.Channel(3)
        self.volume = 0.01
        self.sound.set_volume(self.volume)
        self.hit_sound.set_volume(self.volume)
        self.close_sound.set_volume(self.volume)
        self.crash_sound.set_volume(self.volume)
        #if visual
        self.visual = False
    def new_enemy(self, i=0):
        if i == 0:
            if self.count % 120 == 0:
                self.enemy_list.append(character.enemy0(random.uniform(0, self.WINDOW_WIDTH), 0, 1.5,
                                                        10, True,
                                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 200, (255, 0, 0), 10))
        if i == 1:
            if self.count % 300 == 0:
                self.enemy_list.append(character.enemy1(random.uniform(0, self.WINDOW_WIDTH), 0, 0.5,
                                                        30, True,
                                                        [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 1000, (255, 0, 0), 10))
        if i == 3:
            if self.enemy_list == []:
                self.enemy_list.append(character.boss0(self.WINDOW_WIDTH // 2, 0, 0.5,
                                                       60, True,
                                                       [0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT], 5000, (255, 0, 0), 1))

    def show_heath(self, character, good=False):
        if good:
            pygame.draw.rect(self.window, (0, 255, 0), (0, self.WINDOW_HEIGHT - 5, self.WINDOW_WIDTH * character.health / character.full_health, 5))
        else:
            rect = pygame.Rect(character.position_x - (character.size + 100) / 2, character.position_y - (character.size + 100) / 2, character.size + 100, character.size + 100)
            pygame.draw.arc(self.window, (255, 100, 0), rect, math.radians(+90), math.radians(360 * character.health / character.full_health + 90), 5)

    def draw_image(self, image_path, x, y, size):
        size = size * math.sqrt(2) / 2
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (int(size), int(size)))
        image_rect = image.get_rect(center=(x, y))
        self.window.blit(image, image_rect)

    def action1(self):
        self.learncount+=1
        self.learncount%= 10000
        if self.learntimes%2==0:
            if self.learncount==1:
                #self.learntimes+=1
                agent.save(f'current_best.pth')  # 保存模型状态
            # 1. 如果存在上一次的状态和动作，先进行学习
            if self._last_processed_state and self._last_action:
                # 计算奖励
                reward = 0
                if self.last_time_hit > 0:
                    reward += self.last_time_hit * 1.0  # 命中奖励
                if self.last_time_hurt > 0:
                    reward -= self.last_time_hurt * 100.0  # 受伤惩罚
                # if self.player1.health <= 0:
                #     reward -= 100  # 死亡惩罚
                if len(self.enemy_list) == 0:
                    reward += 100  # 消灭所有敌人奖励

                # 获取当前状态
                current_state, _ = agent._process_game_state(
                    self.enemy_list,
                    [self.player1.position_x, self.player1.position_y],
                    self.last_time_hit,
                    self.last_time_hurt,
                    self.WINDOW_WIDTH,
                    self.WINDOW_HEIGHT
                )

                # 进行Q-learning更新
                agent.learn(reward, current_state, self.player1.health <= 0)

            # 2. 处理当前状态并获取新动作
            processed_state, current_reward = agent._process_game_state(
                self.enemy_list,
                [self.player1.position_x, self.player1.position_y],
                self.last_time_hit,
                self.last_time_hurt,
                self.WINDOW_WIDTH,
                self.WINDOW_HEIGHT
            )

            # 3. 获取动作
            action = agent.get_action(processed_state)

            # 4. 存储当前状态和动作
            self._last_processed_state = processed_state
            self._last_action = action
            self.action = action

            # 5. 重置命中/受伤计数器
            self.last_time_hit = 0
            self.last_time_hurt = 0

    def run(self):
        if self.visual:
            self.channel_sound.play(self.sound)
        while True:
            self.action1()
            direction = 0
            self.count += 1
            if self.count == sys.maxsize:
                self.count = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            shift = 1

            self.player1.shoot(shift)
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

            self.window.fill((0, 0, 0))
            self.new_enemy(self.enemy_type)
            if len(self.enemy_list) > 0:
                remove_en = []
                for i, e in enumerate(self.enemy_list):
                    if e.show:
                        if self.enemy_type == 3:
                            self.show_heath(e, good=False)
                            self.draw_image('src/img.png', e.position_x, e.position_y - 25, e.size + 100)
                        else:
                            pygame.draw.circle(self.window, e.color, (e.position_x, e.position_y), e.size)
            self.player1.update_bullets()

            for i1, i in enumerate(self.player1.bullets):
                if self.count % 2 == 0:
                    x, y = i.before(0, 0.2)
                    color_1 = tuple(max(0, value - 50) for value in i.color)
                    pygame.draw.circle(self.window, color_1, (x, y), i.size)
                if self.count % 3 == 0:
                    x, y = i.before(0, 0.5)
                    color_2 = tuple(max(0, value - 50) for value in i.color)
                    pygame.draw.circle(self.window, color_2, (x, y), i.size)
            #hit检测
            for i1, i in enumerate(self.player1.bullets):
                pygame.draw.circle(self.window, i.color, (i.position_x, i.position_y), i.size)
                for e in self.enemy_list:
                    if math.sqrt((i.position_x - e.position_x) ** 2 + (e.position_y - i.position_y) ** 2) < e.size + i.size and e.show:
                        e.change_health(-i.power)

                        self.last_time_hit+=i.power

                        i.show = False
                        if e.health <= 0 and e.boss:
                            if self.visual:
                                self.channel_crash.play(self.crash_sound)
                    if not self.channel_hit.get_busy():
                        if self.visual:
                            self.channel_hit.play(self.hit_sound)
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
                    for b in e.bullets:
                        if math.sqrt((b.position_x - self.player1.position_x) ** 2 + (b.position_y - self.player1.position_y) ** 2) < b.size + self.player1.size and self.player1.show:

                            self.player1.change_health(-b.power)
                            self.last_time_hurt += b.power

                            b.show = False
                        elif b.size + self.player1.size + 5 > math.sqrt((b.position_x - self.player1.position_x) ** 2 + (b.position_y - self.player1.position_y) ** 2) > b.size + self.player1.size and self.player1.show:
                            if self.visual:
                                self.channel_close.play(self.close_sound)
                        if b.show:
                            pygame.draw.circle(self.window, b.color, (b.position_x, b.position_y), b.size)
                    if e.update(self.player1.position_x, self.player1.position_y) == False:
                        remove_en.append(i)
                for i in sorted(remove_en, reverse=True):
                    del self.enemy_list[i]
            if shift != 1:
                pygame.draw.circle(self.window, self.player1.color, (self.player1.position_x, self.player1.position_y), self.player1.size)
            self.show_heath(self.player1, True)
            #self.clock.tick(1000)
            self.countfps+=1
            #print(agent.bullet_count)
            if self.countfps % 100 == 0:
                print(f'FPS: {self.countfps / (time.time() - self.curtime):.2f}, Bulltets_count: {agent.bullet_count}')
                self.countfps = 0
                self.curtime= time.time()
            if self.visual==False:
                if self.learncount==100:
                    pygame.display.update()
            else:
                pygame.display.update()

# 用于直接运行
if __name__ == "__main__":
    state_space = (9, 20, 3)  # 9 zones, 20 bullets per zone, 3 features per bullet

    agent = STGAgent()
    try:
        agent.load('current_best.pth')  # 尝试加载之前保存的模型
    except FileNotFoundError:
        print("No saved model found, starting fresh.")

    game = Train()
    game.run()