import math
def move_up(record):
    return record[0], record[1] - record[2]*record[3]

def move_up_sin(record):
    return record[0]-record[2]*math.sin(record[3]), record[1] - record[2]*record[3]