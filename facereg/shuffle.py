import random

with open('train-align2.txt') as f:
    lines = f.readlines()

random.shuffle(lines)

with open('train-align2.txt', 'w') as f:
    for l in lines:
        f.write(l)
        
        
with open('test-align2.txt') as f:
    lines = f.readlines()

random.shuffle(lines)

with open('test-align2.txt', 'w') as f:
    for l in lines:
        f.write(l)
