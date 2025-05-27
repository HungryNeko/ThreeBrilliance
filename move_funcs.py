import math
#record=[position_x,position_y,speed,record,p_x,p_y,i]
def move_up(record):
    return record[0], record[1] - record[2]*record[3]

def move_up_sin(record):
    #return record[0]-(record[6]**2)*record[2]*math.sin(record[3]), record[1] - (record[6]**2)*record[2]*record[3]
    return record[0] - (3*record[6] ** 3+record[3]*0.1) * record[2] * math.sin(record[3]), record[1] - record[2]*record[3]

def angle(record,x,y):
    return record[2]*x*record[3]+record[0],record[2]*y*record[3]+record[1]

def angle_degree(record):
    return record[0]+record[4]*record[2]*record[3]*record[6]*0.1,record[1]+record[5]*record[2]*record[3]