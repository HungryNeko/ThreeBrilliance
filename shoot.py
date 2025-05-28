import math
from copy import copy

import objects
import move_funcs
class bullets(objects.objects):
    def __init__(self, position_x=0, position_y=0, speed=5, size=5,
                 check_in=True, range=[0, 0, 400, 800], disappear=10, move_func=1,p_x=0,p_y=0,power=5,i=1,record=0,color=(255,255,255)):
        super().__init__(position_x, position_y, speed, size, check_in, range)
        self.disappear=disappear
        self.move_func=move_func
        self.record=[position_x,position_y,speed,record,p_x,p_y,i]
        self.power=power
        self.show=True
        self.color=color
        self.next_position=[]
        self.last_x = self.record[0]
        self.last_y = self.record[1]
        self.position_x=0
        self.position_y=0


    def move(self,target=None):
        self.last_x=self.record[0]
        self.last_y=self.record[1]
        #print(self.move_func)
        if self.move_func=="up":
            self.position_x,self.position_y=move_funcs.move_up(self.record)
            #print(self.position_x,self.position_y)
        elif self.move_func=="up_sin" :
            self.position_x, self.position_y = move_funcs.move_up_sin(self.record)
        elif self.move_func=="angle_en":
            x1=self.record[0]
            y1=self.record[1]
            x2=self.record[4]
            y2=self.record[5]
            z=math.sqrt((x1-x2)**2+(y1-y2)**2)
            if z==0:
                z=0.01
            x3=(x2-x1)/z
            y3=(y2-y1)/z
            self.position_x, self.position_y = move_funcs.angle(self.record,x3,y3)
        elif self.move_func=='angle_degree':
            self.position_x,self.position_y=move_funcs.angle_degree(self.record)
        else:
            self.position_x, self.position_y = move_funcs.move_up(self.record)
            #print(self.position_x, self.position_y)
        self.record[3]+=1
        if ((self.range[0]-self.disappear<self.position_x<self.range[2]+self.disappear)and(
                self.range[1] - self.disappear < self.position_y < self.range[3] + self.disappear
        ))and self.show:
            #print("r")
            return True
        else:

            return False

    def before(self,target=None,record=0):
        newre=list(self.record)
        newre[3]=newre[3]-record
        #print(newre[3],self.record)
        #print(self.move_func)
        if self.move_func=="up":
            return move_funcs.move_up(newre)
            #print(self.position_x,self.position_y)
        elif self.move_func=="up_sin" :
            return move_funcs.move_up_sin(newre)
        elif self.move_func=="angle_en":
            x1=self.record[0]
            y1=self.record[1]
            x2=self.record[4]
            y2=self.record[5]
            z=math.sqrt((x1-x2)**2+(y1-y2)**2)
            x3=(x2-x1)/z
            y3=(y2-y1)/z
            return move_funcs.angle(newre,x3,y3)
        elif self.move_func=='angle_degree':
            return move_funcs.angle_degree(newre)
        else:
            return move_funcs.move_up(newre)
            #print(self.position_x, self.position_y)
