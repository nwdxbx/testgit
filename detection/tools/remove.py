import os
import numpy as np
import shutil

src = '/media/f/src_data/Face/ssd-face/tmp'
src_txt = '/media/f/src_data/Face/ssd-face/tmp_txt'

txtfiles = os.listdir(src_txt)
src_jpegs= os.listdir(src)
for txtname in txtfiles:
    imgname = txtname[:-3]+'jpg'
    if imgname not in src_jpegs:
        os.remove(os.path.join(src_txt,txtname))

print 'finish...'