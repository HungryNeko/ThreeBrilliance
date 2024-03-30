import objects
import move_funcs
class bullets(objects.objects):
    def __init__(self, position_x=0, position_y=0, speed=5, size=5,
                 check_in=True, range=[0, 0, 400, 800], dissapear=10, move_func=1):
        super().__init__(position_x, position_y, speed, size, check_in, range)
        self.dissapear=dissapear
        self.move_func=move_func
        self.record=0
        self.record=[position_x,position_y,speed,self.record]

    def move(self):
        #print(self.move_func)
        if self.move_func=="up":
            self.position_x,self.position_y=move_funcs.move_up(self.record)
            #print(self.position_x,self.position_y)
        elif self.move_func=="up_sin" :
            self.position_x, self.position_y = move_funcs.move_up_sin(self.record)
        else:
            self.position_x, self.position_y = move_funcs.move_up(self.record)
            #print(self.position_x, self.position_y)
        self.record[3]+=1
        if (self.range[0]-self.dissapear<self.position_x<self.range[2]+self.dissapear)and(
                self.range[1] - self.dissapear < self.position_y < self.range[3] + self.dissapear
        ):
            #print("r")
            return True
        else:
            return False