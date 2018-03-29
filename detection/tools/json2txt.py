import os 
import json
import numpy as np

srcdir = '/media/f/src_data/Face/ssd-face/json'
dstdir = '/media/f/src_data/Face/ssd-face/tmp_txt'

alljsons = os.listdir(srcdir)

for filename in alljsons:
    txtname = filename[:-8] + 'txt'
    json_dict = json.loads(open(os.path.join(srcdir,filename)).read())
    poslist = json_dict['objects']
    fw = open(os.path.join(dstdir,txtname),'w')
    for pos in poslist:
        rect = pos['rect']
        x = int(rect[0])
        y = int(rect[1])
        w = int(rect[2])
        h = int(rect[3])
        result = str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        fw.writelines(result)
    fw.close()

