# PaGAN
This code is for the reviewers of ICDM 2021 submission PaGAN: Generative Adversarial Network for Patent understanding

## Install dependencies
Install dependencies using ```bash install_dependencies.sh``` in https://hub.docker.com/layers/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-devel/images/sha256-ccebb46f954b1d32a4700aaeae0e24bd68653f92c6f276a608bf592b660b63d7?context=explore .

Download Standford-core-nlp for tokenization here https://drive.google.com/file/d/1HS8fq67q9o-mnx4U-JICBIbH7xwsIZeq/view?usp=sharing and place it in src/preproccessing

**Important**: For encoding a text longer than 512 tokens, for example 1500. Set max_pos to 1500 during both preprocessing and training.

You have to rerun code after first download of resources.

For multi-gpu mode you may have to modify line 689 of modeling_bert.py file from pytorch_transformers library from    *extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)*   to    *extended_attention_mask = extended_attention_mask.float()* to make it work.

## Download data
Download contradictions_dataset https://drive.google.com/file/d/1cy3fSMyfIEjOrj2XpVOv2jOosKryai-1/view?usp=sharing and unlabelled_patents https://drive.google.com/file/d/1So98t1hk-gSEbQWr-nns8MXJN1z-n6No/view?usp=sharing and unzip in data_patents/input_data/training_data.

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

Use -num_split to indicate index of fold in k-fold cross-validation.

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

| Model  | Loss  | TP  | FP  | TN  | FN  | Accuracy | Precision  | Recall  | F1 score  | S | S$_m$  |
| SummaTRIZ$_{D}$ | 0.140 | 0     | 0     | 61959 | 2276  | 0.96     | 0 | 0 | 0 | 548   | 1158 |
| SummaTRIZ$_{TL}$ | 0.115 | 576   | 510   | 61449 | 1700  | 0.97     | 0.53     | 0.25     | 0.34     | 1119  | 1711 |
| Baseline$_{ANN_D}$ | 0.140 | 0 | 1 | 65248 | 2276 | 0.97 | 0 | 0 | 0 | 535 | 1149   |
| Baseline$_{ANN_{TL}}$ | 0.115 | 575 | 482 | 64767 | 1701 | 0.97 | 0.54 | 0.25 | 0.35 | 1098 | 1710 |
| \approach$_{PROB}$ | \textbf{0.112} | 575 | 457 | 71584 | 1701 | 0.97 | 0.56 | 0.25 | 0.35 | 1168 | 1736 |
| \approach$_{ANN}$ | \textbf{0.112} | 532 | 414 | 71689 | 1744 | 0.97 | 0.56 | 0.23 | 0.33 | \textbf{1187} | \textbf{1760} |
| \approach$_{LSTM}$ | \textbf{0.112} | 509 | 368 | 71704 | 1767 | 0.97 | \textbf{0.58} | 0.22 | 0.32 | 1186 | 1752 |
| \approach$_{TF}$ | 0.113 | 649 | 592 | 71438 | 1627 | 0.97 | 0.52 | \textbf{0.29} | \textbf{0.37} | 1143 | 1759 |

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
