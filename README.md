# PaGAN
This code is for the reviewers of KR 2021 submission PaGAN: Generative Adversarial Network for Patent understanding

## Install dependencies
Install dependencies using ```bash install_dependencies.sh```

Download Standford-core-nlp for tokenization here https://drive.google.com/file/d/1HS8fq67q9o-mnx4U-JICBIbH7xwsIZeq/view?usp=sharing and place it in src/preproccessing

**Important**: For encoding a text longer than 512 tokens, for example 1500. Set max_pos to 1500 during both preprocessing and training.

Codes are borrowed from PreSumm (https://github.com/nlpyang/PreSumm.git) and ONMT(https://github.com/OpenNMT/OpenNMT-py).

You have to rerun code after first download of resources.

For multi-gpu mode you may have to modify line 689 of modeling_bert.py file from pytorch_transformers library from    *extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)*   to    *extended_attention_mask = extended_attention_mask.float()* to make it work.

## Download data
Download contradictions_dataset https://drive.google.com/file/d/1cy3fSMyfIEjOrj2XpVOv2jOosKryai-1/view?usp=sharing and unlabelled_patents https://drive.google.com/file/d/1So98t1hk-gSEbQWr-nns8MXJN1z-n6No/view?usp=sharing and unzip in data_patents/input_data/training_data.

## Train mode // Patents

Set visibles_gpus to -1 if no gpu available.

**Preprocessing only from patents directories**

*Prepare data for cross validation*:
```
python3 sort.py -real_ratio 0.9 -num_split 1
```
Use -real_ratio to indicate the percentage of labelled data.

Use -num_split to indicate index of fold in k-fold cross-validation.

*Preprocess*:
```
python3 train.py -mode train -parts_of_interest 'STATE_OF_THE_ART' -max_pos 1500 -need_preprocessing True -dataset 'train' 'valid' -real_ratio 1 -num_split 1 -only_preprocessing True
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

*To reproduce results*:
```
bash evaluate.sh
```

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


Pretrained model: https://drive.google.com/file/d/1-UyBnNsbagHJLEA9AJoUM1cWHOXWCn16/view?usp=sharing (you should put it in models directory)
