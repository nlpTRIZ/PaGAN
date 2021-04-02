import os
import shutil

from database.dbPatent import dbPatent


def load_patents(nbr_patents, nbr_years=15):

	database = dbPatent('csip','csip')
	nbr_patents_year = int(nbr_patents/nbr_years)

	shutil.rmtree('../data_patents/input_data/test_data/')
	os.mkdir('../data_patents/input_data/test_data/')

	for i in range(nbr_years):
		year = str(2020-i)
		print(year)
		nbr_patents_years = nbr_patents_year
		if i==0:
			nbr_patents_years+=nbr_patents%nbr_years
		list_patents = database.get_patents(nb = nbr_patents_years, part = None, date = year)

		for current_patent_index in range(len(list_patents)):
			ref = list_patents[current_patent_index]['REF PATENT']
			name_directory = '../data_patents/input_data/test_data/'+ref+'/'


			os.mkdir(name_directory)
			for key in list_patents[current_patent_index].keys():
			    if key!='_id' and key!='REF PATENT':
			        name_file = name_directory+ref+'.'+key
			        text = list_patents[current_patent_index][key]

			        with open(name_file, 'w', encoding='utf-8') as file:
			            if key == 'STATE_OF_THE_ART' and isinstance(text, list):
			                text = text[0]
			            elif isinstance(text, list):
			                text = '\n'.join(text)
			            file.write(text)

			name_file = name_directory+ref+'.'+'SUM'
			with open(name_file, 'w', encoding='utf-8') as file:
			    text='empty'
			    file.write(text)