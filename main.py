import pygame
import sys
import character

# 初始化 Pygame
pygame.init()

# 设置游戏窗口大小
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 800
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("ThreeBrilliance")

# 设置角色的初始位置
player1=character.player(WINDOW_WIDTH//2,WINDOW_HEIGHT//2,
                         10,5,True,
                         [0,0,WINDOW_WIDTH,WINDOW_HEIGHT],100)


# 游戏主循环
clock = pygame.time.Clock()
while True:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 获取按键状态
    keys = pygame.key.get_pressed()
    shift=1
    if keys[pygame.K_LSHIFT]:
        shift=0.5
    # 根据按键状态移动角色
    if keys[pygame.K_LEFT]:
        player1.move_x(-shift)
    if keys[pygame.K_RIGHT]:
        player1.move_x(shift)
    if keys[pygame.K_UP]:
        player1.move_y(-shift)
    if keys[pygame.K_DOWN]:
        player1.move_y(shift)

    if keys[pygame.K_z]:
        player1.shoot()


    # 清除屏幕
    window.fill((0, 0, 0))

    #弹幕
    player1.update_bullets()
    for i in player1.bullets:
        pygame.draw.rect(window, (255, 255, 255), (i.position_x-i.size//2,
                                                   i.position_y-i.size//2, i.size, i.size))

    # 绘制角色
    pygame.draw.rect(window, (255, 0, 0), (player1.position_x - player1.size // 2,
                                           player1.position_y - player1.size // 2, player1.size, player1.size))
    clock.tick(60)
    # 刷新屏幕
    pygame.display.update()
