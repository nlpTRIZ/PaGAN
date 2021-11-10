# PaGAN
This code is for ICDM 2021 paper: **PaGAN: Generative Adversarial Network for Patent understanding**

## Install dependencies
* Install dependencies using ```bash install_dependencies.sh``` in https://hub.docker.com/layers/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-devel/images/sha256-ccebb46f954b1d32a4700aaeae0e24bd68653f92c6f276a608bf592b660b63d7?context=explore .

* Download Standford-core-nlp for tokenization here https://drive.google.com/file/d/1HS8fq67q9o-mnx4U-JICBIbH7xwsIZeq/view?usp=sharing and place it in src/preproccessing

* You have to rerun code after first download of resources.

* For multi-gpu mode you may have to modify line 689 of modeling_bert.py file from pytorch_transformers library from    *extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)*   to    *extended_attention_mask = extended_attention_mask.float()* to make it work.

## Download data
Download contradictions_dataset https://drive.google.com/file/d/1cy3fSMyfIEjOrj2XpVOv2jOosKryai-1/view?usp=sharing and unlabelled_patents https://drive.google.com/file/d/1So98t1hk-gSEbQWr-nns8MXJN1z-n6No/view?usp=sharing and unzip in data_patents/input_data/training_data.

# Details on the dataset and labeling process
State-of-the-art parts of patents from the United States Patent and Trademark Office (USPTO) are used. The dataset contains 1600 states of the art with at least one contradiction and 1600 states of the art without. The length of the patent states of the art is variable but often less than 1500 tokens.

A sentence-level analysis was performed, a contradiction is therefore a pair of sentences containing the parameters forming a contradiction. The sentences of the state of the art can thus belong to 3 different classes. The sentences belonging to the class **first part of contradiction** contain the improved parameter. The class **second part of contradiction** gathers the sentences that contain the degraded parameter when the parameter from the **first part of contradiction** is improved. Finally, a reject class is used to indicate that a sentence does not contain any contradiction information.

The patents were labeled by a team of human experts. The inter-expert variability was reduced with an overall check of all labels by two other experts. Since a patent does not necessarily contain a contradiction, approximately 20,000 patents were analyzed to extract the 1600 patents containing contradictions.

The annotation's policy consists of extracting only one contradiction per patent. If there are several, only the most important one is retained. For each part of a contradiction, all similar sentences (i.e. containing the parameters of the contradiction) are annotated. Thus, in a patent, several possible pairs of sentences can describe the contradiction.

**Example of contradiction from patent US6938300B2:**
**First part of contradiction**: When the stroller 1 moves over a lawn or uneven road surfaces, it is necessary for the stroller wheels to have a large diameter so as to ensure the comfort of the baby. 
**Second part of contradiction**: However, if each of the front wheel assemblies 11 has two large-diameter front wheels 13, the total volume and weight of the stroller 1 will increase significantly so that it is difficult to push the stroller 1.

## Pretrained model 
https://drive.google.com/file/d/1-UyBnNsbagHJLEA9AJoUM1cWHOXWCn16/view?usp=sharing (to be placed in models directory)

This model was trained with num_split=1 (see explanations for preprocessing part)

## Train mode // Patents
Download Transfer Learning model here https://drive.google.com/file/d/1YIo8pp9JCe_4azKVJYVNqxXhOyyYDnnR/view?usp=sharing (to be placed in models directory)

Set visible_gpus to -1 if no gpu available.

**Preprocessing only from patents directories**

*Prepare data for cross validation*:
```
python3 sort.py -real_ratio 0.9 -num_split 1
```
Use -real_ratio to indicate the percentage of labelled data.

Use -num_split to indicate index of fold in 4-fold cross-validation (0,1,2,3).

*Preprocess*:
```
python3 train.py -mode train -parts_of_interest 'STATE_OF_THE_ART' -max_pos 1500 -need_preprocessing True -dataset 'train' 'valid' -real_ratio 0.9 -num_split 1 -only_preprocessing True
```
Use -parts_of_interest option to set parts of the patents to be preprocessed.

Use -dataset option to indicate which input directory will be used.

**Finetuning from pretrained model**
```
python3 train.py -mode train -gan_mode False -parts_of_interest 'STATE_OF_THE_ART' -lr 1e-5 -visible_gpus 0 -train_steps 5 -max_pos 1500 -finetune_bert False -need_preprocessing False -classification 'separate' -model 'bert_sum' -g_learning_rate 1e-5 -batch_size 4500 -doc_classifier LSTM -dataset 'train' 'valid' -real_ratio 0.9 -num_split 1
```
Use -baseline to reproduce baseline results ('SummaTRIZ' or 'baseline').

**Adversarial Training**
```
python3 train.py -mode train -parts_of_interest 'STATE_OF_THE_ART' -lr 1e-5 -visible_gpus 0,1,2,3 -train_steps 5 -max_pos 1500 -finetune_bert True -need_preprocessing False -classification 'separate' -g_learning_rate 1e-5 -batch_size 4 -test_batch_size 40 -generator LSTM_sent -doc_classifier FC -dataset 'train' 'valid' -real_ratio 0.9 -num_split 1 -evaluate_x_steps 20
```

**To reproduce results**:
```
bash evaluate.sh
```
*Note that the following results were obtained at the best of three trainings.*


> Sentence classification 1 (with LSTM generator)

| Model 								              | Loss  	  | TP  	| FP  	| TN  	| FN  	| Accuracy 	| Precision | Recall	  | F1 score  | S 		    | S<sub>m</sub> |
| -----  								              | ----- 	  | ----- | ----- | ----- | ----- | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 		    |
| SummaTRIZ<sub>D</sub> 				      | 0.140 	  | 0     | 0     | 61959 | 2276  | 0.96     	| 0 		    | 0 		    | 0 		    | 548   	  | 1158 			    |
| SummaTRIZ<sub>TL</sub> 				      | 0.115 	  | 576   | 510   | 61449 | 1700  | 0.97     	| 0.53     	| 0.25     	| 0.34     	| 1119  	  | 1711 			    |
| Baseline<sub>ANN<sub>D</sub></sub> 	| 0.140 	  | 0 	  | 1 	  | 65248 | 2276 	| 0.97 		  | 0 		    | 0 		    | 0 		    | 535 		  | 1149   		    |
| Baseline<sub>ANN<sub>TL</sub></sub> | 0.115 	  | 575 	| 482 	| 64767 | 1701 	| 0.97 		  | 0.54 		  | 0.25 		  | 0.35 		  | 1098 		  | 1710 			    |
| PaGAN<sub>PROB</sub> 				        | **0.112** | 575 	| 457 	| 71584 | 1701 	| 0.97 		  | 0.56 		  | 0.25 		  | 0.35 		  | 1168 		  | 1736 			    |
| PaGAN<sub>ANN</sub> 					      | **0.112** | 532 	| 414 	| 71689 | 1744 	| 0.97 		  | 0.56 		  | 0.23 		  | 0.33 		  | **1187** 	| **1760** 		  |
| PaGAN<sub>LSTM</sub> 				        | **0.112** | 509 	| 368 	| 71704 | 1767 	| 0.97 		  | **0.58** 	| 0.22 		  | 0.32 		  | 1186 		  | 1752 			    |
| PaGAN<sub>TF</sub> 					        | 0.113 	  | 649 	| 592 	| 71438 | 1627 	| 0.97 		  | 0.52 		  | **0.29** 	| **0.37** 	| 1143 		  | 1759 			    |


> Sentence classification 2 (with LSTM generator)

| Model 								              | Loss  	  | TP  	| FP  	| TN  	| FN  	| Accuracy 	| Precision | Recall	  | F1 score  | S 		    | S<sub>m</sub> |
| -----  								              | ----- 	  | ----- | ----- | ----- | ----- | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 		    |
| SummaTRIZ<sub>D</sub>               | 0.171     | 0     | 0     | 60526 | 3709  | 0.94      | 0         | 0         | 0         | 1750      | 2692          |
| SummaTRIZ<sub>TL</sub>              | 0.129     | 1814  | 815   | 59711 | 1895  | 0.96      | **0.69**  | 0.49      | 0.57      | 2493      | 3127          |
| Baseline<sub>ANN<sub>D</sub></sub>  | 0.170     | 0     | 0     | 63816 | 3709  | 0.95      | 0         | 0         | 0         | 1766      | 2692          |
| Baseline<sub>ANN<sub>TL</sub></sub> | 0.129     | 1849  | 881   | 62935 | 1860  | 0.96      | 0.68      | 0.50      | 0.57      | 2500      | 3131          |
| PaGAN<sub>PROB</sub>                | 0.120     | 2091  | 936   | 69672 | 1618  | **0.97**  | **0.69**  | 0.56      | 0.62      | 2619      | **3226**      | 
| PaGAN<sub>ANN</sub>                 | 0.120     | 2288  | 1119  | 69551 | 1421  | **0.97**  | 0.67      | 0.62      | 0.64      | 2626      | 3206          |
| PaGAN<sub>LSTM</sub>                | **0.118** | 2323  | 1156  | 69483 | 1386  | **0.97**  | 0.67      | **0.63**  | **0.65**  | **2645**  | 3213          |
| PaGAN<sub>TF</sub>                  | 0.121     | 2327  | 1228  | 69369 | 1382  | 0.96      | 0.65      | **0.63**  | 0.64      | 2631      | 3220          |


> Document classification (with LSTM generator)

| Model 								              | Loss  	  | TP  	| FP  	| TN  	| FN  	| Accuracy 	| Precision | Recall	  | F1 score  | S 		    | S<sub>m</sub> |
| -----  								              | ----- 	  | ----- | ----- | ----- | ----- | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 	  | ----- 		    |
| SummaTRIZ<sub>D</sub>               | -         | 0     |0      | 1600  | 1600  | 0.50      | 0         | 0         | 0         | 153       | 0             |
| SummaTRIZ<sub>TL</sub>              | -         | 386   | 135   | 1465  | 1214  | 0.58      | 0.74      | 0.24      | 0.36      | 582       | 213           |
| Baseline<sub>ANN<sub>D</sub></sub>  | 0.529     | 1274  | 490   | 1110  | 326   | 0.74      | 0.72      | 0.80      | 0.76      | 146       | 96            |
| Baseline<sub>ANN<sub>TL</sub></sub> | 0.502     | 1275  | 438   | 1162  | 325   | 0.76      | 0.74      | 0.80      | 0.77      | 580       | 467           |
| PaGAN<sub>PROB</sub>                | -         | 335   | 126   | 1474  | 1265  | 0.57      | 0.73      | 0.21      | 0.33      | **668**   | 192           |
| PaGAN<sub>ANN</sub>                 | **0.466** | 1335  | 431   | 1169  | 265   | 0.78      | **0.76**  | 0.83      | 0.79      | 666       | **576**       |
| PaGAN<sub>LSTM</sub>                | 0.481     | 1370  | 507   | 1093  | 230   | 0.77      | 0.73      | **0.86**  | 0.79      | 654       | 567           |
| PaGAN<sub>TF</sub>                  | 0.467     | 1345  | 427   | 1173  | 255   | **0.79**  | **0.76**  | 0.84      | **0.80**  | 648       | 552           |


> Sentence classification 1 (with ANN document classifier)

| Setup Generator     | Loss      | TP    | FP    | TN    | FN    | Acc.    | Pre.      | Recall    | F1 score  | CO<sub>Found</sub>  | CO<sub>valid</sub>  |
| -----  						  | ----- 	  | ----- | ----- | ----- | ----- | ----- 	| ----- 	  | ----- 	  | ----- 	  | ----- 	            | ----- 		          |
| LSTM<sub>S          | 0.112     | 532   | 414   | 71689 | 1744  | 0.97    | 0.56      | 0.23      | 0.33      | **1187**            | **1760**            |
| FC<sub>S            | 0.113     | 575   | 493   | 71540 | 1701  | 0.97    | 0.54      | **0.25**  | **0.34**  | 1151                | **1760**            |
| TF<sub>S            | 0.116     | 313   | 199   | 71873 | 1963  | 0.97    | 0.61      | 0.14      | 0.22      | 1157                | 1741                |
| LSTM<sub>ALL</sub>  | 0.114     | 561   | 438   | 71672 | 1715  | 0.97    | 0.56      | **0.25**  | **0.34**  | 1146                | 1729                |
| TF<sub>ALL</sub>    | 0.118     | 286   | 170   | 71863 | 1990  | 0.97    | **0.63**  | 0.13      | 0.21      | 1132                | 1714                |
| LSTM<sub>D</sub>    | 0.112     | 462   | 326   | 71771 | 1814  | 0.97    | 0.59      | 0.20      | 0.30      | 1168                | 1736                |
| TF<sub>D</sub>      | **0.111** | 537   | 387   | 71730 | 1739  | 0.97    | 0.58      | 0.24      | **0.34**  | 1182                | 1759                |


> Sentence classification 2 (with ANN document classifier)

| Setup Generator     | Loss      | TP    | FP    | TN    | FN    | Acc.    | Pre.      | Recall    | F1 score  | CO<sub>Found</sub>  | CO<sub>valid</sub>  |
| -----  							| ----- 	  | ----- | ----- | ----- | ----- | ----- 	| ----- 	  | ----- 	  | ----- 	  | ----- 	            | ----- 		          |
| LSTM<sub>S          | 0.120     | 2288  | 1119  | 69551 | 1421  | 0.97    | 0.67      | **0.62**  | **0.64**  | 2626                | 3206                |
| FC<sub>S            | **0.118** | 2285  | 1135  | 69465 | 1424  | 0.97    | 0.67      | **0.62**  | **0.64**  | 2630                | 3231                |
| TF<sub>S            | 0.130     | 1735  | 690   | 69949 | 1974  | 0.96    | 0.72      | 0.47      | 0.57      | 2602                | 3192                |
| LSTM<sub>ALL</sub>  | 0.121     | 2285  | 1153  | 69524 | 1424  | 0.97    | 0.66      | **0.62**  | 0.64      | 2639                | 3217                |
| TF<sub>ALL</sub>    | 0.129     | 1593  | 587   | 70013 | 2116  | 0.96    | **0.73**  | 0.43      | 0.54      | 2598                | 3185                |
| LSTM<sub>D</sub>    | **0.118** | 1960  | 773   | 69891 | 1749  | 0.97    | 0.72      | 0.53      | 0.61      | **2663**            | **3245**            |
| TF<sub>D</sub>      | **0.118** | 2269  | 1060  | 69624 | 1440  | 0.97    | 0.68      | 0.61      | **0.64**  | 2637                | 3231                |

  
> Document classification (with ANN document classifier)

| Setup Generator     | Loss      | TP    | FP    | TN    | FN    | Acc.    | Pre.      | Recall    | F1 score  | CO<sub>Found</sub>  | CO<sub>valid</sub>  |
| -----  							| ----- 	  | ----- | ----- | ----- | ----- | ----- 	| ----- 	  | ----- 	  | ----- 	  | ----- 	            | ----- 		          |
| LSTM<sub>S          | 0.466     | 1335  | 431   | 1169  | 265   | 0.78    | **0.76**  | 0.83      | 0.79      | **666**             | **576**             |
| FC<sub>S            | **0.459** | 1363  | 466   | 1134  | 237   | 0.78    | 0.75      | 0.85      | 0.79      | 632                 | 548                 |
| TF<sub>S            | 0.508     | 1363  | 532   | 1068  | 237   | 0.76    | 0.72      | 0.85      | 0.78      | 630                 | 547                 |
| LSTM<sub>ALL</sub>  | 0.460     | 1384  | 467   | 1133  | 216   | **0.79**| 0.75      | 0.86      | **0.80**  | 620                 | 531                 |
| TF<sub>ALL</sub>    | 0.512     | 1449  | 618   | 982   | 151   | 0.76    | 0.70      | **0.91**  | 0.79      | 598                 | 552                 |
| LSTM<sub>D</sub>    | 0.471     | 1392  | 498   | 1102  | 208   | 0.78    | 0.74      | 0.87      | **0.80**  | 641                 | 575                 |
| TF<sub>D</sub>      | 0.479     | 1418  | 592   | 1008  | 182   | 0.76    | 0.71      | 0.89      | 0.79      | 653                 | 574                 |

## Test mode // Patents
From directory '../data_patents/input_data/test_data/' or other by changing option -input_directory:
```
python3 train.py -mode test -visible_gpus -1 -input 'txt' -depth_directory 0 -length_summary 1 -load_model ../models/PaGAN_pretrained.pt -need_preprocessing True
```

With files' path:
```
python3 train.py -mode test -input_files '../data_patents/input_data/test_data/sample.txt' -visible_gpus -1 -length_summary 1 -load_model ../models/PaGAN_pretrained.pt -need_preprocessing True
```

Use -length_summary option to set the number of output sentences per part / per file

Use -test_threshold option to set the minimum probability of summary sentences


Test generates a csv file in ../results with list of dictionnary containing summary sentences, their probability, patents' reference etc...
