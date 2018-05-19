import os
root_dir = "/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/112x112"
save_file_name = "/media/f/src_data/Face/FaceRecognitionDataSets/megface_src/112x112/lst"

with open(save_file_name, 'w') as f:
	for parent, dirnames, filenames in os.walk(root_dir):
		for filename in filenames:
		    if filename[-1] == "g" or filename[-1] == "G":
			    real_path = "1" + "\t" + os.path.join(parent, filename) + "\t" + "0"
			    f.write(real_path + "\n")
