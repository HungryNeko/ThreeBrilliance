import math

import pygame
import sys

# 初始化 Pygame
pygame.init()

# 设置窗口大小
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("测试显示分割出的图像")


def load_character_spritesheet(image_path, rows, cols, width, height, re=0, resize=None):
    # 加载图像
    image = pygame.image.load(image_path)
    if resize is not None:
        resize = resize * math.sqrt(2) / 2
    # 计算每个子表面的宽度和高度
    # 创建一个空列表来存储所有的子表面
    sprites = []
    # 逐行逐列获取子表面，并添加到列表中
    for row in range(rows):
        for col in range(cols):
            rect = pygame.Rect((width + re) * col, (height + re) * row, width, height)
            subsurface = image.subsurface(rect)
            if resize is not None:
                subsurface = pygame.transform.scale(subsurface, (resize,resize))
            sprites.append(subsurface)
    return sprites


def main():
    # 加载包含多个人物素材的图片，并指定行数和列数
    spritesheet = load_character_spritesheet("src/img_1.png", 4, 3,50,50)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 清除屏幕
        screen.fill((255, 255, 255))

        # 绘制所有分割出的子表面
        x, y = 0, 0
        for sprite in spritesheet:
            screen.blit(sprite, (0, 0))
            x += sprite.get_width()  # 移动到下一个位置
            if x >= WINDOW_WIDTH:
                x = 0
                y += sprite.get_height()  # 移动到下一行

        # 刷新屏幕
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
