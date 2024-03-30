class objects():
    def __init__(self, position_x=0, position_y=0, speed=5, size=5,
                 check_in=True, range=[0, 0, 400, 800]):
        self.position_x = position_x
        self.position_y = position_y
        self.size = size
        self.speed = speed
        self.check_in = check_in
        self.range = range

    def move_x(self, positive=1, x=None):
        if not x:
            x = self.speed
        x = x * positive
        if self.check_in:
            if self.range[0] <= self.position_x + x <= self.range[2]:
                self.position_x += x
        else:
            self.position_x += x

    def move_y(self, positive=1, y=None):
        if not y:
            y = self.speed
        y = y * positive
        if self.check_in:
            if self.range[1] <= self.position_y + y <= self.range[3]:
                self.position_y += y
        else:
            self.position_y += y

class character(objects):
    def __init__(self, position_x, position_y, speed, size, check_in, range,health=100):
        super().__init__(position_x, position_y, speed, size, check_in, range)
        self.health=health
        self.bullets=[]

    def change_health(self, change):
        self.health += change