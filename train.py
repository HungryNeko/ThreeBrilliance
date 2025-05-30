import math
import random
import sys
import time
import pygame
from sympy.physics.units import action

import character
import img
from RLmodel import STGAgent


class Train:
    def __init__(self):
        self.boss_exist = False
        self.last_time_hit=0
        self.rand=0
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
        self.enemy_type = 0
        self.learncount=0
        self.learntimes=0
        self.stage=0
        self.stage_kill=0
        self.stage_kill_boss=0

        # 资源加载
        self.spritesheet = img.load_character_spritesheet("src/img_1.png", 4, 3, 50, 50)
        self.close_sound = pygame.mixer.Sound("src/东方原作音效/绀长擦弹.wav")
        self.hit_sound = pygame.mixer.Sound("src/东方原作音效/莎莎火箭弹命中.wav")
        self.crash_sound = pygame.mixer.Sound("src/东方原作音效/击破boss.wav")
        self.sound = pygame.mixer.Sound("src/th15_13.mp3")
        #self.sound.play(-1)  # 循环播放背景音乐
        self.channel_sound = pygame.mixer.Channel(0)
        self.channel_hit = pygame.mixer.Channel(1)
        self.channel_close = pygame.mixer.Channel(2)
        self.channel_crash = pygame.mixer.Channel(3)
        self.volume = 0.5
        self.sound.set_volume(self.volume)
        self.hit_sound.set_volume(self.volume)
        self.close_sound.set_volume(self.volume)
        self.crash_sound.set_volume(self.volume)
        #if visual
        self.visual = True

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
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (int(size), int(size)))
        image_rect = image.get_rect(center=(x, y))
        self.window.blit(image, image_rect)

    def action1(self):
        self.learncount+=1
        self.learncount%= 10000
        if self.learntimes%1==0:
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
                agent.learn(reward, current_state, self._last_action)

            # 2. 处理当前状态并获取新动作
            processed_state, current_reward = agent._process_game_state(
                self.enemy_list,
                [self.player1.position_x, self.player1.position_y],
                self.last_time_hit,
                self.last_time_hurt,
                self.WINDOW_WIDTH,
                self.WINDOW_HEIGHT
            )
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
        if self.visual:
            #self.channel_sound.play(self.sound)
            pass
        #time.sleep(10)
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
            print(self.stage,self.stage_kill_boss,self.stage_kill)
            if self.stage==3 and self.boss_exist==False and self.stage_kill_boss>0:
                self.enemy_list=[]
            if self.stage==0 and self.stage_kill>10 and not self.boss_exist:
                self.stage=1
                self.enemy_type=1
                self.stage_kill_boss = 0
                self.stage_kill=0
            elif self.stage==1 and self.stage_kill>10 and not self.boss_exist:
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
                            self.show_heath(e, good=False)
                            if self.stage==5:
                                e.position_x,e.position_y=self.move_randomly(e.position_x,e.position_y,e.speed*1.5)
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
                        if e.health<=0:
                            self.stage_kill += 1
                            if e.boss:
                                self.stage_kill_boss+=1
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
                    if e.boss:
                        for b in e.bullets:
                            b.record[2]*=0.9999
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
            #self.draw_agent_zones(self.window, agent, (self.player1.position_x, self.player1.position_y))
            self.show_heath(self.player1, True)
            self.clock.tick(60)
            self.countfps+=1
            #print(agent.bullet_count)
            if self.countfps % 100 == 0:
                #print(f'FPS: {self.countfps / (time.time() - self.curtime)}:.2f, Qtable is not new: {agent.q_table!={}}')
                self.countfps = 0
                self.curtime= time.time()
            if self.visual==False:
                if self.learncount==100:
                    pygame.display.update()
            else:
                pygame.display.update()

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
# 用于直接运行
if __name__ == "__main__":
    state_space = (9, 20, 3)  # 9 zones, 20 bullets per zone, 3 features per bullet

    agent = STGAgent()

    try:
        pass
        agent.load('current_best.pth')  # 尝试加载之前保存的模型
    except FileNotFoundError:
        print("No saved model found, starting fresh.")

    game = Train()
    game.run()