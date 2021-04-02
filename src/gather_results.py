import json
import glob

all_results = sorted(glob.glob("../results/results*.txt"))
all_types = sorted(list(set([name.split('_')[-3]+'+'+name.split('_')[-5]+name.split('_')[-4] for name in all_results])))
print("Results for: "+' '.join(all_types))

gathered_results=[{} for _ in all_types]
num_files=[ 0 for _ in all_types]

for file in all_results:
	num_dic=all_types.index(file.split('_')[-3]+'+'+file.split('_')[-5]+file.split('_')[-4])
	num_files[num_dic]+=1
	f = open(file,) 
	data = json.load(f)
	results= {'sentence_level':{'Loss_s':data['D_losses']['D_sent_supervised_loss'][0],
		'Loss_s1':data['D_losses']['D_sent_all_loss'][0],
		'TP1':data['TP'][0],
		'FP1':data['FP'][0],
		'TN1':data['TN'][0],
		'FN1':data['FN'][0],
		'Loss_s2':data['D_losses']['D_sent_all_loss'][1],
		'TP2':data['TP'][1],
		'FP2':data['FP'][1],
		'TN2':data['TN'][1],
		'FN2':data['FN'][1],
		'Found':data['Found sentences'],
		'Found_margin':data['Found sentences with margin']},
		'doc_level':{'Loss_d':data['D_losses']['D_co_supervised_loss'][0],
		'TP':data['G_prediction_for_G_extraction'][0]+\
			data['G_prediction_for_W_extraction'][1]-\
			data['G_prediction_for_W_extraction'][0],
		'FP':400-data['G_prediction_for_no_possible_extraction'][0],
		'TN':data['G_prediction_for_no_possible_extraction'][0],
		'FN':400-(data['G_prediction_for_G_extraction'][0]+\
			data['G_prediction_for_W_extraction'][1]-\
			data['G_prediction_for_W_extraction'][0]),
		'Found':data['Found contradictions'][0],
		'Valid':data['G_prediction_for_G_extraction'][0]
		}}
	if not gathered_results[num_dic]:
		gathered_results[num_dic]=results
	else:
		for level in gathered_results[num_dic].keys():
			for key in gathered_results[num_dic][level].keys():
				# print(results[level][key])
				if not isinstance(results[level][key], list): 
					gathered_results[num_dic][level][key]+=results[level][key]
				else:
					for i,elmt in enumerate(results[level][key]):
						gathered_results[num_dic][level][key][i]+=elmt

#mean over losses
for dic, norm in zip(gathered_results,num_files):
	dic['sentence_level']['Loss_s']/=norm
	dic['sentence_level']['Loss_s1']/=norm
	dic['sentence_level']['Loss_s2']/=norm
	dic['doc_level']['Loss_d']/=norm/10
	for level in dic.keys():
		if level=='sentence_level':
			# first part
			dic[level]['Acc1']=(dic[level]['TP1']+dic[level]['TN1'])/\
				(dic[level]['TP1']+dic[level]['TN1']+dic[level]['FP1']+dic[level]['FN1'])
			if dic[level]['TP1']+dic[level]['FP1']>0:
				dic[level]['Prec1']=(dic[level]['TP1'])/\
					(dic[level]['TP1']+dic[level]['FP1'])
			else:
				dic[level]['Prec1']=0
			dic[level]['Recall1']=(dic[level]['TP1'])/\
				(dic[level]['TP1']+dic[level]['FN1'])
			if dic[level]['Prec1']+dic[level]['Recall1']>0:
				dic[level]['F11']=2*dic[level]['Prec1']*dic[level]['Recall1']/\
					(dic[level]['Prec1']+dic[level]['Recall1'])
			else:
				dic[level]['F11']=0
			#second part
			dic[level]['Acc2']=(dic[level]['TP2']+dic[level]['TN2'])/\
				(dic[level]['TP2']+dic[level]['TN2']+dic[level]['FP2']+dic[level]['FN2'])
			if dic[level]['TP2']+dic[level]['FP2']>0:
				dic[level]['Prec2']=(dic[level]['TP2'])/\
					(dic[level]['TP2']+dic[level]['FP2'])
			else:
				dic[level]['Prec2']=0
			dic[level]['Recall2']=(dic[level]['TP2'])/\
				(dic[level]['TP2']+dic[level]['FN2'])
			if dic[level]['Prec2']+dic[level]['Recall2']>0:
				dic[level]['F12']=2*dic[level]['Prec2']*dic[level]['Recall2']/\
					(dic[level]['Prec2']+dic[level]['Recall2'])
			else:
				dic[level]['F12']=0
		else:
			dic[level]['Acc']=(dic[level]['TP']+dic[level]['TN'])/\
				(dic[level]['TP']+dic[level]['TN']+dic[level]['FP']+dic[level]['FN'])
			dic[level]['Prec']=(dic[level]['TP'])/\
				(dic[level]['TP']+dic[level]['FP'])
			dic[level]['Recall']=(dic[level]['TP'])/\
				(dic[level]['TP']+dic[level]['FN'])
			if dic[level]['Prec']+dic[level]['Recall']>0:
				dic[level]['F1']=2*dic[level]['Prec']*dic[level]['Recall']/\
					(dic[level]['Prec']+dic[level]['Recall'])
			else:
				dic[level]['F1']=0

#generate table
for level in gathered_results[0].keys():
	print('\n'+level.upper())
	if level=='sentence_level':
		print("\nModel%-35s & %-5s & %-5s & %-5s & %-5s & %-5s & %-8s & %-8s & %-8s & %-8s & %-5s & %-7s  \\\\"%('','Loss','TP','FP','TN','FN','Acc','Prec','Recall','F1','S', 'S$_m$'))
	else:
		print("\nModel%-35s & %-5s & %-5s & %-5s & %-5s & %-5s & %-8s & %-8s & %-8s & %-8s & %-8s & %-8s \\\\"%('','Loss','TP','FP','TN','FN','Acc','Prec','Recall','F1','Found CO','Valid CO'))
	if level=='doc_level':
		for dic,name in zip(gathered_results,all_types):
			print("Model$_{%-30s}$ & %.3f & %-5d & %-5d & %-5d & %-5d & %.2f     & %.2f     & %.2f     & %.2f     & %-5d    & %-5d    \\\\"%(
				name, dic[level]['Loss_d'], dic[level]['TP'],dic[level]['FP'],dic[level]['TN'],dic[level]['FN'],dic[level]['Acc'],dic[level]['Prec'],dic[level]['Recall'],dic[level]['F1'],dic[level]['Found'],dic[level]['Valid']))
	else:
		for part in range(2):
			for dic,name in zip(gathered_results,all_types):
				if part==0:
					print("Model$_{%-30s}$ & %.3f & %-5d & %-5d & %-5d & %-5d & %.2f     & %.2f     & %.2f     & %.2f     & %-5d & %-5d    \\\\"%(
						name, dic[level]['Loss_s1'], dic[level]['TP1'],dic[level]['FP1'],dic[level]['TN1'],dic[level]['FN1'],dic[level]['Acc1'],dic[level]['Prec1'],dic[level]['Recall1'],dic[level]['F11'],dic[level]['Found'][0],dic[level]['Found_margin'][0]))
				else:
					print("Model$_{%-30s}$ & %.3f & %-5d & %-5d & %-5d & %-5d & %.2f     & %.2f     & %.2f     & %.2f     & %-5d & %-5d    \\\\"%(	
						name, dic[level]['Loss_s2'], dic[level]['TP2'],dic[level]['FP2'],dic[level]['TN2'],dic[level]['FN2'],dic[level]['Acc2'],dic[level]['Prec2'],dic[level]['Recall2'],dic[level]['F12'],dic[level]['Found'][1],dic[level]['Found_margin'][1]))
