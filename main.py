import math
import random

import pygame
import sys
import img
from pygame import font


import character

# 初始化 Pygame
pygame.init()
AUDIO_ENABLED = False


class NullSound:
    def set_volume(self, volume):
        pass


class NullChannel:
    def play(self, sound):
        pass

    def get_busy(self):
        return False


def safe_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except (FileNotFoundError, pygame.error):
        return NullSound()


def safe_channel(index):
    try:
        return pygame.mixer.Channel(index)
    except pygame.error:
        return NullChannel()

# 设置游戏窗口大小
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 900

def initial_ui_scale():
    info = pygame.display.Info()
    if info.current_w <= 0 or info.current_h <= 0:
        return 0.8
    width_fit = info.current_w * 0.9 / WINDOW_WIDTH
    height_fit = info.current_h * 0.82 / WINDOW_HEIGHT
    return max(0.35, min(1.0, width_fit, height_fit))

def scaled_size(scale):
    return (max(1, int(WINDOW_WIDTH * scale)), max(1, int(WINDOW_HEIGHT * scale)))

ui_scale = initial_ui_scale()
display_size = scaled_size(ui_scale)
display = pygame.display.set_mode(display_size, pygame.RESIZABLE)
window = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT)).convert()
pygame.display.set_caption("ThreeBrilliance")

# 设置角色的初始位置
player1=character.player(WINDOW_WIDTH//2,WINDOW_HEIGHT//2,
                         10,5,True,
                         [0,0,WINDOW_WIDTH,WINDOW_HEIGHT-5],100,(20, 255, 255))


# 游戏主循环
clock = pygame.time.Clock()
enemy_list=[]
count=0
enemy_type=3

def set_display_size(size):
    global display, display_size, ui_scale
    width = max(240, int(size[0]))
    height = max(360, int(size[1]))
    display_size = (width, height)
    ui_scale = min(width / WINDOW_WIDTH, height / WINDOW_HEIGHT)
    display = pygame.display.set_mode(display_size, pygame.RESIZABLE)

def change_ui_scale(delta):
    global display, display_size, ui_scale
    ui_scale = max(0.35, min(1.5, ui_scale + delta))
    display_size = scaled_size(ui_scale)
    display = pygame.display.set_mode(display_size, pygame.RESIZABLE)

def present():
    scaled_frame = pygame.transform.smoothscale(window, scaled_size(ui_scale))
    display.fill((0, 0, 0))
    offset_x = (display_size[0] - scaled_frame.get_width()) // 2
    offset_y = (display_size[1] - scaled_frame.get_height()) // 2
    display.blit(scaled_frame, (offset_x, offset_y))
    pygame.display.update()

def handle_event(event):
    global ui_scale, display_size, display
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
    if event.type == pygame.VIDEORESIZE:
        set_display_size((event.w, event.h))
        return
    if event.type != pygame.KEYDOWN:
        return
    if event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
        change_ui_scale(0.1)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        change_ui_scale(-0.1)
    elif event.key in (pygame.K_0, pygame.K_KP0):
        ui_scale = initial_ui_scale()
        display_size = scaled_size(ui_scale)
        display = pygame.display.set_mode(display_size, pygame.RESIZABLE)

def new_enemy(list,c,i=0):
    if i==0:
        if c%120==0:
            list.append(character.enemy0(random.uniform(0,WINDOW_WIDTH),0,1.5,
                                         10,True,
                                         [0,0,WINDOW_WIDTH,WINDOW_HEIGHT],200,(255,0,0),10))
    if i==1:
        if c%300==0:
            list.append(character.enemy1(random.uniform(0,WINDOW_WIDTH),0,0.5,
                                         30,True,
                                         [0,0,WINDOW_WIDTH,WINDOW_HEIGHT],1000,(255,0,0),10))
    if i==3:
        if list==[]:
            list.append(character.boss0(WINDOW_WIDTH//2,0,0.5,
                                         60,True,
                                         [0,0,WINDOW_WIDTH,WINDOW_HEIGHT],5000,(255, 0, 0),1,
                                         volume=volume))

def show_heath(character,good=False):
    #print(character.full_health,character.health)
    if good:
        pygame.draw.rect(window,(0,255,0),(0,WINDOW_HEIGHT-5,WINDOW_WIDTH*character.health/character.full_health,5))
    else:
        rect = pygame.Rect(character.position_x -(character.size+100)/2, character.position_y - (character.size+100)/2, character.size+100, character.size+100)
        pygame.draw.arc(window, (255,100,0), rect, math.radians(+90), math.radians(360*character.health/character.full_health+90), 5)

def draw_image(image_path, x, y, size):
    size=size*math.sqrt(2)/2
    try:
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (int(size), int(size)))
    except (FileNotFoundError, pygame.error):
        image = pygame.Surface((int(size), int(size)), pygame.SRCALPHA)
        center = int(size) // 2
        radius = max(8, int(size * 0.32))
        pygame.draw.circle(image, (220, 60, 90), (center, center), radius)
        pygame.draw.circle(image, (255, 210, 80), (center, center), radius, 3)
        pygame.draw.line(image, (255, 255, 255), (center - radius, center), (center + radius, center), 2)
        pygame.draw.line(image, (255, 255, 255), (center, center - radius), (center, center + radius), 2)
    image_rect = image.get_rect(center=(x, y))
    window.blit(image, image_rect)


def rungame():
    if AUDIO_ENABLED:
        channel_sound.play(sound)
    global count
    while True:
        direction=0
        count+=1
        if count==sys.maxsize:
            count=0
        # 处理事件
        for event in pygame.event.get():
            handle_event(event)

        # 获取按键状态
        keys = pygame.key.get_pressed()
        shift=1
        if keys[pygame.K_LSHIFT]:
            shift=0.3
        # 根据按键状态移动角色
        if keys[pygame.K_LEFT]:
            direction = 1
            player1.move_x(-shift)
        if keys[pygame.K_RIGHT]:
            direction = 2
            player1.move_x(shift)
        if keys[pygame.K_UP]:
            player1.move_y(-shift)
        if keys[pygame.K_DOWN]:
            player1.move_y(shift)

        if keys[pygame.K_z]:
            player1.shoot(shift)


        # 清除屏幕
        window.fill((0, 0, 0))
        #新敌人，绘制敌人
        new_enemy(enemy_list,count,enemy_type)
        if len(enemy_list)>0:
            remove_en=[]
            for i,e in enumerate(enemy_list):
                #绘制敌人
                if e.show:

                    #boss 血条
                    if enemy_type==3:
                        show_heath(e, good=False)
                        draw_image('src/img.png',e.position_x,e.position_y-25,e.size+100)
                        #print(e.health)

                    else:
                        pygame.draw.circle(window, e.color, (e.position_x, e.position_y), e.size)
        #弹幕
        player1.update_bullets()
        #玩家子弹
        for i1,i in enumerate(player1.bullets):

            if count%2==0:
                x,y=i.before(0,0.2)
                color_1= tuple(max(0, value - 50) for value in i.color)
                pygame.draw.circle(window, color_1, (x,
                                                           y), i.size)
            if count%3==0:
                x, y = i.before(0,0.5)
                color_2 = tuple(max(0, value - 50) for value in i.color)
                pygame.draw.circle(window, color_2, (x,
                                                          y), i.size)
        for i1, i in enumerate(player1.bullets):
            pygame.draw.circle(window, i.color, (i.position_x,
                                                       i.position_y), i.size)
            #攻击敌人
            for e in enemy_list:
                if math.sqrt((i.position_x-e.position_x)**2+(e.position_y-i.position_y)**2)<e.size+i.size and e.show:#命中
                    e.change_health(-i.power)
                    i.show=False
                    if e.health <= 0 and e.boss:
                        # print(10)
                        if AUDIO_ENABLED:
                            channel_crash.play(crash_sound)
                        # print("1")
                if AUDIO_ENABLED and not channel_hit.get_busy():
                    channel_hit.play(hit_sound)
                    #print(channel_hit.get_busy())
        # 绘制角色
        pl = count // 10
        if direction == 0:
            window.blit(spritesheet[pl % 3], (player1.position_x - 25-1, player1.position_y - 25))
        elif direction == 2:
            window.blit(spritesheet[pl % 3 + 3], (player1.position_x - 25-1, player1.position_y - 25))
        else:
            window.blit(spritesheet[pl % 3 + 6], (player1.position_x - 25-1, player1.position_y - 25))

        #敌人生产
        if len(enemy_list)>0:
            for i,e in enumerate(enemy_list):
                e.update_bullets()
                #敌人弹幕
                for b in e.bullets:
                    if math.sqrt((b.position_x - player1.position_x) ** 2 + (
                            b.position_y - player1.position_y) ** 2) < b.size + player1.size and player1.show:#受伤
                        player1.change_health(-b.power)
                        b.show=False
                    elif b.size + player1.size+5>math.sqrt((b.position_x - player1.position_x) ** 2 + (#擦弹
                            b.position_y - player1.position_y) ** 2) > b.size + player1.size and player1.show:
                        #print("!")
                        if AUDIO_ENABLED:
                            channel_close.play(close_sound)
                    if b.show:
                        pygame.draw.circle(window, b.color, (b.position_x,
                                                               b.position_y), b.size)
                    #b.move()
                if e.update(player1.position_x,player1.position_y)==False:
                    #print("delete", i)
                    remove_en.append(i)
            for i in sorted(remove_en, reverse=True):

                del enemy_list[i]


        if shift!=1:
            pygame.draw.circle(window, player1.color, (player1.position_x,
                                                   player1.position_y), player1.size)

        show_heath(player1,True)

        clock.tick(60)
        # 刷新屏幕
        present()

spritesheet = img.load_character_spritesheet("src/img_1.png", 4, 3,50,50)
close_sound = safe_sound("src/东方原作音效/绀长擦弹.wav")
hit_sound=safe_sound("src/东方原作音效/莎莎火箭弹命中.wav")
crash_sound=safe_sound("src/东方原作音效/击破boss.wav")
sound=safe_sound("src/th15_13.mp3")
channel_sound = safe_channel(0)
channel_hit = safe_channel(1)
channel_close = safe_channel(2)
channel_crash = safe_channel(3)
volume=0.01 if AUDIO_ENABLED else 0.0
sound.set_volume(volume)
hit_sound.set_volume(volume)
close_sound.set_volume(volume)
crash_sound.set_volume(volume)
rungame()
