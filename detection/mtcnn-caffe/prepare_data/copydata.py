import os
def two_pos():
    ori_path = "24backup"
    new_path = "24"
    ori_file = open(os.path.join(ori_path, 'pos_24.txt'),'r')
    new_file = open(os.path.join(new_path, 'pos_24.txt'),'r')
    save_file = open('pos_24.txt','w')
    ori_lists = ori_file.readlines()
    for ori_list in ori_lists:
        save_file.write(ori_list)
    ori_file.close()
    new_lists = new_file.readlines()
    for new_list in new_lists:
        save_file.write(new_list)
    new_file.close()
    save_file.close()

def cls_list():
    ori_file = open('/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/24/pos_24.txt','r')
    ori_file_2 = open('/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/24/neg_24.txt','r')
    save_file = open('./24/cls_24.txt','w')
    ori_lists = ori_file.readlines()
    for ori_list in ori_lists:
        save_file.write(ori_list)
    ori_file.close()
    new_lists = ori_file_2.readlines()
    for new_list in new_lists:
        save_file.write(new_list)
    save_file.close()

def roi_list():
    ori_file = open('/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/24/pos_24.txt','r')
    ori_file_2 = open('/media/ffh/f85082d9-0526-4d9a-89b4-61594dbaa936/mtcnn/data/change/24/part_24.txt','r')
    save_file = open('./24/roi_24.txt','w')
    ori_lists = ori_file.readlines()
    for ori_list in ori_lists:
        save_file.write(ori_list)
    ori_file.close()
    new_lists = ori_file_2.readlines()
    for new_list in new_lists:
        save_file.write(new_list)
    save_file.close()
 
if __name__ == '__main__':
    #two_pos()
    cls_list()
    roi_list()