import csv
import re
import shutil
import os

with open('../data_patents/input_data/Brevets_arcelor_mittal.csv', mode='r', encoding="utf8", errors='ignore') as csv_file:
	name_dir = '../data_patents/input_data/test_arcelor/'
	csv_reader = csv.reader(csv_file, delimiter=';')
	for i,row in enumerate(csv_reader):
		if i>0 and row[0]:
			print(row[0])
			name_dir_patent = name_dir+row[0]+'/'
			try:
				shutil.rmtree(name_dir_patent)
				print("Deleting directory...")
			except:
				pass
			print(name_dir_patent)
			os.mkdir(name_dir_patent)
			with open(name_dir_patent+row[0]+'.STATE_OF_THE_ART', 'w', encoding='utf-8') as file:
				content = re.sub(r"[0-9]{4,4}|\[|\]|\n|\([0-9]{,4}[a-z]{,1}\)",'',' '.join(row[2::]))
				file.write(content)
			with open(name_dir_patent+row[0]+'.REF', 'w', encoding='utf-8') as file:
				file.write(row[0])
			with open(name_dir_patent+row[0]+'.ABSTRACT', 'w', encoding='utf-8') as file:
				file.write(row[1])