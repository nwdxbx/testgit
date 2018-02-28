# -*- coding: UTF-8 -*-  
import xml.dom.minidom

dom = xml.dom.minidom.parse('/media/d/datasets/ssd_python/test/Img_Video_公司_Bandicam_4.xml'.decode('utf8'))
root = dom.documentElement
res = root.getElementsByTagName('object')
for r in res:
    x = r.getElementsByTagName('xmin')[0]
    print(x.firstChild.data)
