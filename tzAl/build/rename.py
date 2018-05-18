import os 
import shutil

srcdir = "/media/e/bat/QT_program/tzAl/build/imgs"
dstdir = "/media/e/bat/QT_program/tzAl/build/dst"
imgs = os.listdir(srcdir)
for img in imgs:
    lst = img.strip().split(' ')
    filename = lst[0] + '_' + lst[1]
    shutil.copyfile(os.path.join(srcdir,img),os.path.join(dstdir,filename))
    