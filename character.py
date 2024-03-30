import objects
import shoot
import move_funcs
class player(objects.character):
    def __init__(self, position_x, position_y, speed, size, check_in, range,health):
        super().__init__(position_x, position_y, speed, size, check_in, range,health)

    def shoot(self):
        self.bullets.append(shoot.bullets(self.position_x,self.position_y,20,5,False,self.range,move_func="up"))
        self.bullets.append(shoot.bullets(self.position_x,self.position_y,20,5,False,self.range,move_func="up_sin"))

    def update_bullets(self):
        remove_list = []
        for i, b in enumerate(self.bullets):
            if b.move() == False:
                #print("remove")
                remove_list.append(i)
        for i in sorted(remove_list, reverse=True):
            del self.bullets[i]