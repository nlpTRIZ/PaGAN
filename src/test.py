#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import argparse
import os
import torch

from others.logging import init_logger
from train_extractive import train_ext, test_ext
from preprocess import Preprocessor

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

os.environ["CLASSPATH"] = "./preprocessing/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Saved models dict
separate_dict = {"classifier":['MLP','MLP'],
                 "model":'../models/model_best_separate_bertsum_distilbert.pt',
                 "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}
# separate_dict_prediction = {"classifier":['MLP','MLP'],
#                             "model":'../models/model_best_separate_bertsum_distilbert_prediction.pt',
                            # "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}
multi_class_dict = {"classifier":['MLP'],
                    "model":'../models/model_best_multiclass_bertsum_distilbert.pt',
                    "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}
both_dict = {"classifier":['MLP'],
             "model":'../models/model_best_both_bertsum_distilbert.pt',
             "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}
first_dict = {"classifier":['MLP'],
              "model":'../models/model_best_first_bertsum_distilbert.pt',
              "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}
second_dict = {"classifier":['MLP'],
               "model":'../models/model_best_second_bertsum_distilbert.pt',
               "embedding_model":['bert_sum','distilbert-base-nli-stsb-mean-tokens']}

models_dict={"separate":separate_dict,"multi_class":multi_class_dict,"both":both_dict,"first":first_dict,"second":second_dict}



parser = argparse.ArgumentParser()


# Preprocessing parameters
########################################################################################################################################################################################################
parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])
parser.add_argument('-need_preprocessing', type=str2bool, nargs='?',const=True,default=True)

# For testing
# Choose if you want to process all files with extension "input" in "input_directory" or a list of files paths using "input files"
parser.add_argument('-input_files',nargs='*', help='list of paths to files to be summarized, for example: "../data_patents/input_data/test.txt" "../data_patents/input_data/test2.txt"', default=None)
parser.add_argument('-input_directory', type=str, default='../data_patents/input_data/dataset/test', help='Directory containing input files')
parser.add_argument('-input', type=str, default=['STATE_OF_THE_ART'], help='Extension of input file')

# For training
parser.add_argument('-data_augmentation', type=str, default="None", help='"None", translation" or "transformation"')
parser.add_argument('-translation_language', type=str, default="ca", help='korean:"ko", chinese:"zh-TW", catalan:"ca"')
parser.add_argument('-transformation_type', type=str, default="bert_embeddings", help='"bert_embeddings", "word2vec_embeddings" or "synonyms"')

parser.add_argument('-dataset_dir', type=str, default='../data_patents/input_data/dataset')
parser.add_argument('-dataset', type=str, default='', help='train, valid or test, default will process all datasets')
parser.add_argument("-save_path_dir", type=str, default='../data_patents/preprocessed_data')
parser.add_argument("-save_path_prepro", type=str, default='../data_patents/preprocessed_data/preprocessing_results_init')
parser.add_argument("-save_path_nn", type=str, default='../data_patents/preprocessed_data/preprocessing_results')

# Summaries parameters
parser.add_argument('-parts_of_interest', help='delimited list input', type=str, default='STATE_OF_THE_ART')
parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')

# Model parameters
parser.add_argument('-min_src_nsents', default=1, type=int)
parser.add_argument('-max_src_nsents', default=1000, type=int)
parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
parser.add_argument('-max_src_ntokens_per_sent', default=1000, type=int)
parser.add_argument('-max_tgt_ntokens', default=1000, type=int)
parser.add_argument('-min_tgt_ntokens', default=1, type=int)
parser.add_argument('-use_bert_basic_tokenizer', default=True, type=str)



# BERT summary parameters
########################################################################################################################################################################################################
parser.add_argument("-model", default=['bert_sum'], nargs='*', choices=['bert_sum', 
                                                            'distilbert-base-nli-stsb-mean-tokens',
                                                            'bert-base-nli-stsb-mean-tokens', 
                                                            'roberta-base-nli-stsb-mean-tokens',
                                                            'both'])
parser.add_argument("-classification", default='separate', type=str, choices=['first','second', 'both', 'multi_class','separate']) #for multi_class, new classes must be added before first and second label
parser.add_argument("-classifier", default=None, nargs='*', choices=['MLP', 'Linear'])
parser.add_argument("-predict_contradiction", default=True, type=str2bool)

# Add the prefix of the part you want to use
parser.add_argument("-bert_data_path", default='../data_patents/preprocessed_data/preprocessing_results/STATE_OF_THE_ART')

parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/cnndm')
parser.add_argument("-temp_dir", default='../temp')

parser.add_argument("-use_contradiction_only", default=False, type=str2bool)
parser.add_argument("-batch_size", default=6000, type=int)

# Test parameters
parser.add_argument("-test_batch_size", default=30000, type=int)
parser.add_argument("-test_threshold", default=None, type=float)
parser.add_argument("-length_summary", default=None, type=int)


parser.add_argument("-max_pos", default=1500, type=int)
parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-dec_layers", default=6, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)

# params for EXT
parser.add_argument("-ext_dropout", default=0.2, type=float)
parser.add_argument("-ext_layers", default=2, type=int)
parser.add_argument("-ext_hidden_size", default=768, type=int)
parser.add_argument("-ext_heads", default=8, type=int)
parser.add_argument("-ext_ff_size", default=2048, type=int)

parser.add_argument("-label_smoothing", default=0.1, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=0, type=int)
parser.add_argument("-max_length", default=1500, type=int)
parser.add_argument("-max_tgt_len", default=500, type=int)


parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

parser.add_argument("-save_checkpoint_steps", default=5, type=int)
parser.add_argument("-accum_count", default=1, type=int)
parser.add_argument("-report_every", default=1, type=int)
parser.add_argument("-train_steps", default=1000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/patents.log')
parser.add_argument('-seed', default=666, type=int)

# parser.add_argument("-load_model", default='')
# parser.add_argument("-cnn_daily_mail_model", type=str2bool, nargs='?',const=True,default=True)

parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

parser.add_argument('-n_cpus', default=1, type=int)

args = parser.parse_args()


def apply_model(input_files, queue=None, predict_contradiction=False, visible_gpus='3', length_summary=20):
    
    # Added for function
    #####################################
    args.mode = 'test'
    args.input_files = [input_files]
    args.visible_gpus = visible_gpus
    args.length_summary = length_summary
    args.predict_contradiction = predict_contradiction
    #####################################

    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    args.cnn_daily_mail_model=False

    args.load_model=models_dict[args.classification]["model"]
    args.classifier=models_dict[args.classification]["classifier"]
    args.model=models_dict[args.classification]["embedding_model"]
    if args.predict_contradiction:
        args.load_model='../models/model_best_separate_bertsum_distilbert_prediction.pt'

    if args.need_preprocessing:
        Preprocesseur=Preprocessor(args)
        Preprocesseur.tokenize()
        Preprocesseur.format_to_lines()
        Preprocesseur.format_to_nn()

    cp = args.load_model
    step = 0

    if args.predict_contradiction:
        for i,(summaries,output_probas, prediction_contradiction) in enumerate(test_ext(args, device_id, cp, step)):
            if i==0:
                results_summaries = summaries
                result_probas = output_probas
                result_pred = prediction_contradiction
            else:
                results_summaries+=summaries
                result_probas+=output_probas
                result_pred+=prediction_contradiction
        queue.put([output_probas,summaries, prediction_contradiction])
    else:
        for i,(summaries,output_probas) in enumerate(test_ext(args, device_id, cp, step)):
            if i==0:
                results_summaries = summaries
                result_probas = output_probas
            else:
                results_summaries+=summaries
                result_probas+=output_probas
        queue.put([output_probas,summaries])
        
    
    torch.cuda.empty_cache()

