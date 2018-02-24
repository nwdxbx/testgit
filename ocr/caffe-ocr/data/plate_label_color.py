#!/usr/bin/python
#-*-coding:utf-8 -*-
import os
import cv2
import re

plate = []


dict = {'0': '0', '1': 1, '2': '2', '3': '3', '4': 4, '5': '5', '6': '6', '7': 7, '8': '8', '9': '9',
	'A': '10', 'B': 11, 'C': '12', 'D': '13', 'E': 14, 'F': '15', 'G': '16', 'H': 17, 'J': '18', 'K': '19', 'L': 20, 'M': '21', 'N': '22', 'P': '23', 'Q': '24',
        'R': '25', 'S': '26', 'T': '27', 'U': '28', 'V': '29', 'W': '30', 'X': '31', 'Y': '32', 'Z': '33', "京": "34", "沪": "35", "津": "36", "渝": "37",
        '黑': '38', '吉': '39', '辽': '40', '蒙': '41', '冀': '42', '新': '43', '甘': '44', '青': '45', '陕': '46', '宁': '47', '豫': '48', '鲁': '49',
        '晋': '50', '皖': '51', '鄂': '52', '湘': '53', '苏': '54', '川': '55', '贵': '56', '云': '57', '桂': '58', '藏': '59', '浙': '60', '赣': '61',
        '粤': '62', '闽': '63', '琼': '64', '挂': '65', '学': '66', '警': '67'}

color = {'blue': '0', 'green': '1', 'white': '2', 'black': '3', 'yellow': '4'}
color_index = 0
bDouble = 0


dir_path = "/media/d/ocr/caffe_ocr/data/val"
fd=open('./val.txt','w')

num = 0


for root,dirs,files in os.walk(dir_path):
	for filename in files:
		plate = []
		bDouble = 0
		img_path = os.path.join(root,filename)
		img_path = img_path.strip('\n')
		print "img_path=",img_path

		split_name = re.split('_|\.', filename)
		print "split_name=",split_name
		plate_num = split_name[1]
		plate_len = len(plate_num)
		print "plate_len=",plate_len
		if plate_len < 5:
			print "plate_num=",plate_num
			continue

		ret1 = img_path.find('blue')
		if ret1 != -1:
		    color_index = color['blue']

		ret2 = img_path.find('green')
		if ret2 != -1:
		    color_index = color['green']

		ret3 = img_path.find('white')
		if ret3 != -1:
		    color_index = color['white']

		ret4 = img_path.find('black')
		if ret4 != -1:
		    color_index = color['black']

		ret5 = img_path.find('yellow')
		if ret5 != -1:
		    color_index = color['yellow']

		ret5 = img_path.find('double')
		if ret5 != -1:
		    bDouble = 1

		#print "color_index=",color_index

		#print "plate_num=",plate_num

		province = plate_num[0]
		province = plate_num.decode('utf-8')[0:1].encode('utf-8')
		if plate_len == 11:
			plate.append(dict[plate_num.decode('utf-8')[0:1].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[1:2].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[2:3].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[3:4].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[4:5].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[5:6].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[6:7].encode('utf-8')])
			label_name = img_path+' '+"68 68 68 68 68 68 68 68 68 68 68 68 68 "+str(plate[0])+' '+str(plate[1])+' '+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(plate[5])+' '+str(plate[6])+' '+str(color_index)+'\n'
		elif plate_len == 9:
			plate.append(dict[plate_num.decode('utf-8')[0:1].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[1:2].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[2:3].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[3:4].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[4:5].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[5:6].encode('utf-8')])
			plate.append(dict[plate_num.decode('utf-8')[6:7].encode('utf-8')])
			label_name = img_path+' '+"68 68 68 68 68 68 68 68 68 68 68 68 68 "+str(plate[0])+' '+str(plate[1])+' '+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(plate[5])+' '+str(plate[6])+' '+str(color_index)+'\n'
		elif plate_len == 6:
			plate.append(dict['0'])
			plate.append(dict[plate_num[1]])
			plate.append(dict[plate_num[2]])
			plate.append(dict[plate_num[3]])
			plate.append(dict[plate_num[4]])
			plate.append(dict[plate_num[5]])
			label_name = img_path+' '+"68 68 68 68 68 68 68 68 68 68 68 68 68 "+str(54)+' '+str(0)+' '+str(plate[1])+' '+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(plate[5])+' '+str(color_index)+'\n'
		elif plate_len == 5:
			print "plate_num=",plate_num
			#print plate_num[0]
			#print dict[plate_num[0]]
			plate.append(dict['0'])
			plate.append(dict[plate_num[1]])
			plate.append(dict[plate_num[2]])
			plate.append(dict[plate_num[3]])
			plate.append(dict[plate_num[4]])
			label_name = img_path+' '+"68 68 68 68 68 68 68 68 68 68 68 68 68 "+str(54)+' '+str(0)+' '+str(plate[1])+' '+str(plate[1])+' '+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(color_index)+'\n'
		else:
			print "error: platenum=",plate_num
			continue

		#print "plate_org(%d %c %c %c %c %c %c)"%(54,plate_num[0],plate_num[1],plate_num[2],plate_num[3],plate_num[4],plate_num[5])
		#print("plate({0} {1} {2} {3} {4} {5} {6})".format(plate[0],plate[1],plate[2],plate[3],plate[4],plate[5],plate[6]))
		#if bDouble == 0:
		#	label_name = img_path+' '+"65 65 65 65 65 65 65 65 65 65 65 65 65 "+str(plate[0])+' '+str(plate[1])+' '+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(plate[5])+' '+str(plate[6])+' '+str(color_index)+'\n'
		#else:
		#	label_name = img_path+' '+str(plate[0])+' '+str(plate[1])+' '+"65 65 65 65 65 65 65 65 65 65 65 65 65 "+str(plate[2])+' '+str(plate[3])+' '+str(plate[4])+' '+str(plate[5])+' '+str(plate[6])+' '+str(color_index)+'\n'
		#print "label_name=",label_name
		fd.write(label_name)
		num = num + 1
		print "num=",num

fd.close()
print "========================SUCCESS=============================="


