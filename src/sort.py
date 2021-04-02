import glob
import os
import random
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-real_ratio", default=None, type=float)
parser.add_argument("-num_split", default=None, type=int, choices=[0,1,2,3])
args = parser.parse_args()

try:
	os.mkdir('../data_patents/input_data/training_data/train/')
	os.mkdir('../data_patents/input_data/training_data/valid/')
except:
	shutil.rmtree('../data_patents/input_data/training_data/train/')
	shutil.rmtree('../data_patents/input_data/training_data/valid/')
	os.mkdir('../data_patents/input_data/training_data/train/')
	os.mkdir('../data_patents/input_data/training_data/valid/')

nbr_train = 1200
nbr_valid = 400
ratio_real = args.real_ratio
ratio_real_test = 0.5

labl_pos_files = sorted(glob.glob('../data_patents/input_data/training_data/contradictions_dataset/with/*'))
labl_neg_files = sorted(glob.glob('../data_patents/input_data/training_data/contradictions_dataset/without/*'))
print('%d positives labeled files.' %len(labl_pos_files))
print('%d negatives labeled files.' %len(labl_neg_files))

random.seed(1)
shuffle_lists = list(zip(labl_pos_files,labl_neg_files))
random.shuffle(shuffle_lists)
labl_pos_files, labl_neg_files = zip(*shuffle_lists)

valid_files = []
train_files = []
for list_file in [labl_pos_files,labl_neg_files]:
	valid_files += list_file[args.num_split*400:(args.num_split+1)*400]
	train_files += [file for file in list_file if file not in valid_files]

for data_files, type_file in zip([train_files, valid_files],['train','valid']):
	for file in data_files:
		name_file = os.path.basename(file)
		shutil.copytree(file, '../data_patents/input_data/training_data/'+type_file+'/'+name_file)

for i,path in enumerate(['../data_patents/input_data/training_data/unlabelled_patents/*']):
	files = sorted(glob.glob(path))

	random.seed(2)
	random.shuffle(files)

	nbr_train = int(2*nbr_train/ratio_real*(1-ratio_real))
	nbr_valid = int(2*nbr_valid/ratio_real_test*(1-ratio_real_test))
	# nbr_test = int(2*nbr_test/ratio_real_test*(1-ratio_real_test))
	print("Nbr unlabeled patents for train set: ",nbr_train)
	print("Nbr unlabeled patents for valid set: ",nbr_valid)
	
	ind = 0
	while(ind<nbr_train):
		file=files[ind]
		name_file = os.path.basename(file)
		try:
			shutil.copytree(file, '../data_patents/input_data/training_data/train/'+name_file)
		except:
			print(name_file)
			nbr_train +=1
		ind+=1
	
	start = ind
	while(ind<start+nbr_valid):
		file=files[ind]
		name_file = os.path.basename(file)
		try:
			shutil.copytree(file, '../data_patents/input_data/training_data/valid/'+name_file)
		except:
			print(name_file)
			nbr_valid +=1
		ind+=1

