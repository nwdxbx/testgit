import os
import cv2
import re
import numpy as np
def makexml(xmlfile, imgwh, xywh):
    fid=open(xmlfile, 'w')
    beginstr='''<annotation>
    <folder>VOC2007</folder>
    <filename>20171016_000004_1506040597_142_6.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>329145082</flickrid>
    </source>
    <owner>>
        <flickrid>hiromori2</flickrid>
        <name>Hiroyuki Mori</name>
    </owner>>
    <size>
        <width>%s</width>
        <height>%s</height>
        <depth>%s</depth>
    </size>
    <segmented>0</segmented>
    '''
    midstr='''<object>
       <name>%s</name>
       <pose>Unspecified</pose>
       <truncated>0</truncated>
       <difficult>0</difficult>
       <bndbox>
           <xmin>%d</xmin>
           <ymin>%d</ymin>
           <xmax>%d</xmax>
           <ymax>%d</ymax>
       </bndbox>
    </object>
    '''
    endstr='</annotation>'
    fid.write(beginstr % (imgwh[1], imgwh[0], imgwh[2]))
    for i in xywh:
        # if i[4] == str(1):
        #     label = 'backface'
        # else:
        #     label = 'frontface'
        fid.write(midstr % ('renlian',int(i[0]), int(i[1]), int(i[0])+int(i[2])-1, int(i[1])+int(i[3])-1))
    fid.write(endstr)
    fid.close()
    

txtdir = '/media/f/src_data/Face/ssd-face/front_real_txt'
xmldir = '/media/f/src_data/Face/ssd-face/front_Annotations'
srcdir = '/media/f/src_data/Face/ssd-face/front_JPEGImages'
filtxts = os.listdir(txtdir)
fw = open('test.txt','w')
for filtxt in filtxts:
    f = open(os.path.join(txtdir,filtxt),'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        arrayline = line.strip().split()
        xywhl=[]
        xywhl.append([arrayline[0],arrayline[1],arrayline[2],arrayline[3]])
    imgname = filtxt[:-3] + 'jpg'
    sz =cv2.imread(os.path.join(srcdir,imgname)).shape
    xmlfile = filtxt[:-3] + 'xml'

    makexml(os.path.join(xmldir,xmlfile),sz,xywhl)
    fw.write('%s %s\n' % ('front_JPEGImages/'+imgname,'front_Annotations/'+xmlfile))


# with open('train.txt') as f:
#     lines=f.readlines()
# fid=open('train2.txt', 'w')
# for line in lines:
#     li=re.split(' |\n', line)
#     with open(li[1]) as f:
#         lines2=f.readlines()
#     xywh=[]
#     for i in lines2:
#         li2=re.split(' |\n', i)
#         xywh.append(map(lambda x:float(x), [li2[0], li2[1], li2[2], li2[3]]))
    
#     sz=cv2.imread(li[0]).shape
#     print(sz)
#     xmlfile=li[2]
#     makexml(xmlfile, sz, xywh)
    
#     fid.write('%s %s\n' % (li[0], xmlfile))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
