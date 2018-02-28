import os

with open('trainval.txt') as f:
    lines = f.readlines()
    
    
rootpath = os.path.join(os.getcwd(), 'data')

for l in lines:
    l = l[:-1]
    l = l.split(' ')
    l1 = os.path.join(rootpath, 'JPEGImages', os.path.basename(l[0]))
    l2 = os.path.join(rootpath, 'Annotations', os.path.basename(l[1]))
    
    cmd1 = 'cp %s %s' % (l[0], l1)
    cmd2 = 'cp %s %s' % (l[1], l2)
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
