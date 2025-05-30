import math
import sys

import pygame

import objects
import shoot
import move_funcs
class player(objects.character):
    def __init__(self, position_x, position_y, speed, size, check_in, range,health,color):
        super().__init__(position_x, position_y, speed, size, check_in, range,health,color,)
        self.show = True
        self.count=0
        self.level=0

    def shoot(self,i):
        self.count+=1
        if self.count==sys.maxsize:
            self.count=0

        self.bullets.append(shoot.bullets(self.position_x,self.position_y,30,5,False,self.range,10,"up",0,0,5,i,color=(255, 195,205)))
        if self.level > 0:
            self.bullets.append(shoot.bullets(self.position_x,self.position_y,20,5,False,self.range,500,"up_sin",0,0,5,i,color=(100, 255, 100)))
            if self.level>1:
                self.bullets.append(
                    shoot.bullets(self.position_x, self.position_y, 20, 5, False, self.range, 500, "angle_degree", 1, -1, 5, i,color=(255, 195,205)))
                if self.level > 2:
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 20, 5, False, self.range, 500, "angle_degree", -1, -1, 5, i,color=(255, 195,205)))
                    if self.count%20==0 and self.level>3:
                        self.bullets.append(
                            shoot.bullets(self.position_x+20, self.position_y, 10, 8, False, self.range, 500, "up", 0, 0, 20, i,color=(255, 255,0)))
                        self.bullets.append(
                            shoot.bullets(self.position_x-20, self.position_y, 10, 8, False, self.range, 500, "up", 0, 0, 20, i,color=(255, 255,0)))


class enemy0(objects.character):
    def __init__(self, position_x, position_y, speed, size, check_in, range, health,color,fire_rate=60):
        super().__init__(position_x, position_y, speed, size, check_in, range, health,color)
        self.show=True
        self.count=0
        self.fire_rate=fire_rate
        self.boss=False
    def shoot(self,x,y):
        if self.show==True:
            self.bullets.append(shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10,"angle_en",x,y,5))
        else:
            return False
    def move(self):
        self.position_y=self.position_y+self.speed
        #print(self.position_y,self.speed)
        if(self.range[1]-10<self.position_y<self.range[3]+10):
            return True
        else:

            return False

    def update(self,x,y):

        #print("update enemy")
        self.count+=1
        if self.count==sys.maxsize:
            self.count=0
        if self.count%self.fire_rate==0:
            self.shoot(x,y)
        if self.move()==False:
            #print(False)
            self.show=False
        if self.health<0:
            self.show=False

        if self.show==False and len(self.bullets)==0:
            #print(False)
            return False
        else:
            return True

class enemy1(enemy0):
    def __init__(self, position_x, position_y, speed, size, check_in, range, health,color,fire_rate):
        super().__init__(position_x, position_y, speed, size, check_in, range, health,color,fire_rate)

    def shoot(self,x,y):
        if self.show==True:
            self.bullets.append(shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10,"angle_en",x,y,5))
            self.bullets.append(
                shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10, "angle_en", x+10, y, 5))
            self.bullets.append(
                shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10, "angle_en", x, y+10, 5))
        else:
            return False
    def move(self):
        self.position_y=self.position_y+self.speed
        #print(self.position_y,self.speed)
        if(self.range[1]-10<self.position_y<self.range[3]+10):
            return True
        else:

            return False

class boss1(enemy0):
    def __init__(self, position_x, position_y, speed, size, check_in, range, health,color,fire_rate,volume=0.5):
        super().__init__(position_x, position_y, speed, size, check_in, range, health,color,fire_rate)

        self.boss_card_sound = pygame.mixer.Sound("src/东方原作音效/弹幕展开tan.wav")
        self.boss_card_sound.set_volume(volume)
        self.channel_card=pygame.mixer.Channel(4)
        self.boss=True

    def shoot(self,x,y):
        if self.show==True:
            if self.count%3==0:
                for i in range(-5,6):#旋转
                    #self.bullets.append(shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10,"angle_en",x+i*100,y,5))
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 3, 5, False, self.range, 20, "angle_en",
                                      self.position_x + math.sin(i * 36+self.count/self.fire_rate/20), self.position_y+math.cos(i * 36+self.count/self.fire_rate/20), 5))
            for i in range(-50, 60):#画圈
                if self.count % 80 == 0:
                    self.channel_card.play(self.boss_card_sound)
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10, "angle_en",
                                      self.position_x + math.sin(i * 7.2 + self.count / self.fire_rate / 5),
                                      self.position_y + math.cos(i * 7.2 + self.count / self.fire_rate / 5), 5,color=(255,165,0)))
            if self.count%8==0:#反向旋转
                for i in range(-2, 1):
                    # self.bullets.append(shoot.bullets(self.position_x, self.position_y, 5, 5, False, self.range, 10,"angle_en",x+i*100,y,5))
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 3, 5, False, self.range, 10, "angle_en",
                                      self.position_x - math.sin(i * 90 - self.count / self.fire_rate / 300),
                                      self.position_y - math.cos(i * 90 - self.count / self.fire_rate / 300), 5,color=(255,165,0)))
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 3, 5, False, self.range, 10, "angle_en",
                                      self.position_x - math.sin(i * 90 - self.count / self.fire_rate / 300+30),
                                      self.position_y - math.cos(i * 90 - self.count / self.fire_rate / 300+30), 5,
                                      color=(255, 165, 0)))
        else:
            return False
    def move(self):
        #print(self.position_y,self.range[1]*0.3,self.range)
        if self.position_y<self.range[3]*0.3:
            self.position_y=self.position_y+self.speed
        #print(self.position_y,self.speed)
        if(self.range[1]-10<self.position_y<self.range[3]+10):
            return True
        else:
            return False

class boss0(enemy0):
    def __init__(self, position_x, position_y, speed, size, check_in, range, health,color,fire_rate,volume=0.5):
        super().__init__(position_x, position_y, speed, size, check_in, range, health,color,fire_rate)

        self.boss_card_sound = pygame.mixer.Sound("src/东方原作音效/弹幕展开tan.wav")
        self.boss_card_sound.set_volume(volume)
        self.channel_card=pygame.mixer.Channel(4)
        self.boss=True

    def shoot(self,x,y):
        if self.show==True:
            for i in range(-70, 80):#画圈
                if self.count % 50 == 0:
                    self.channel_card.play(self.boss_card_sound)
                    self.bullets.append(
                        shoot.bullets(self.position_x, self.position_y, 10, 5, False, self.range, 10, "angle_en",
                                      self.position_x + math.sin(i * 7.2 + self.count / self.fire_rate / 5),
                                      self.position_y + math.cos(i * 7.2 + self.count / self.fire_rate / 5), 5,color=(255,165,0)))

        else:
            return False
    def move(self):
        #print(self.position_y,self.range[1]*0.3,self.range)
        if self.position_y<self.range[3]*0.3:
            self.position_y=self.position_y+self.speed
        #print(self.position_y,self.speed)
        if(self.range[1]-10<self.position_y<self.range[3]+10):
            return True
        else:
            return False