#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import argparse
import os
import csv
import gc
import logging
import sys

import torch

from others.logging import init_logger
from train_extractive import train_ext, test_ext
from preprocess import Preprocessor

os.environ["CLASSPATH"] = "./preprocessing/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar"


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data/cross validation options
    parser.add_argument('-real_ratio', default=1, type=float)
    parser.add_argument('-num_split', default=0, type=int)

    # Preprocessing parameters
    ########################################################################################################################################################################################################
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-need_preprocessing', type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('-only_preprocessing', type=str2bool, nargs='?',const=True,default=False)

    # For testing
    # Choose if you want to process all files with extension "input" in "input_directory" or a list of files paths using "input files"
    parser.add_argument('-input_files',nargs='*', help='list of paths to files to be summarized, for example: "../data_patents/input_data/test.txt" "../data_patents/input_data/test2.txt"', default=None)
    parser.add_argument('-input_directory', type=str, default='../data_patents/input_data/test_data/', help='Directory containing input files')
    parser.add_argument('-depth_directory', type=int, default=0, help='number of sub_directories between input_directory and data files')
    parser.add_argument('-input', type=str, default='STATE_OF_THE_ART', help='Extension of input file')
    parser.add_argument('-random_test_patents', type=int, default=None)

    # For training
    parser.add_argument('-data_augmentation', type=str, default="None", help='"None", translation" or "transformation"')
    parser.add_argument('-translation_language', type=str, default="ca", help='korean:"ko", chinese:"zh-TW", catalan:"ca"')
    parser.add_argument('-transformation_type', type=str, default="bert_embeddings", help='"bert_embeddings", "word2vec_embeddings" or "synonyms"')
    parser.add_argument('-dataset_dir', type=str, default='../data_patents/input_data/training_data')
    parser.add_argument('-dataset', nargs='*', default=['train','valid','test'], help='train, valid or test, default will process all datasets')
    parser.add_argument("-save_path_dir", type=str, default='../data_patents/preprocessed_data')
    parser.add_argument("-save_path_prepro", type=str, default='../data_patents/preprocessed_data/preprocessing_results_init')
    parser.add_argument("-save_path_nn", type=str, default='../data_patents/preprocessed_data/preprocessing_results')
    
    # Summaries parameters
    parser.add_argument('-parts_of_interest', help='delimited list input', type=str, default='STATE_OF_THE_ART')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')

    # Input parameters
    parser.add_argument('-min_src_nsents', default=1, type=int)
    parser.add_argument('-max_src_nsents', default=1000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=1000, type=int)
    parser.add_argument('-max_tgt_ntokens', default=1000, type=int)
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)
    parser.add_argument('-use_bert_basic_tokenizer', default=True, type=str)

    # PaGAN model parameters
    ########################################################################################################################################################################################################
    parser.add_argument('-baseline', default=None, type=str, choices=['SummaTRIZ','baseline'])
    parser.add_argument('-transfer_learning', default=True, type=str2bool)
    parser.add_argument("-parallel_computing", default=True, type=str2bool)
    parser.add_argument('-load_model', default=None, type=str)
    parser.add_argument('-load_generator', default=None, type=str)
    parser.add_argument("-classification", default='separate', type=str, choices=['first','second', 'both', 'multi_class','separate']) #for multi_class, new classes must be added before first and second label
    parser.add_argument("-doc_classifier", default='Transformer', type=str, choices=['LSTM', 'GRU', 'Transformer', 'FC', 'FC_all', 'Probabilistic'])
    parser.add_argument('-nbr_class_neurons', default=2, type=int)
    parser.add_argument('-evaluate_x_steps', default=-1, type=int)
    parser.add_argument("-use_packed_data", default=False, type=str2bool)
    parser.add_argument("-weights_separate_loss", nargs='*',default=[2,1], type=int)
    parser.add_argument('-coeff_model_losses', nargs='*', type=float, default=[0,1/10,1,1,0,1])

    # Add the prefix of the part you want to use
    parser.add_argument("-bert_data_path", default='../data_patents/preprocessed_data/preprocessing_results/STATE_OF_THE_ART')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-use_contradiction_only", default=False, type=str2bool)
    parser.add_argument("-batch_size", default=6000, type=int)

    # Test parameters
    parser.add_argument("-test_batch_size", default=64, type=int)
    parser.add_argument("-test_threshold", default=None, type=float)
    parser.add_argument("-length_summary", default=None, type=int)
    parser.add_argument("-nbr_contradictions", default=1, type=int)
    
    parser.add_argument("-max_pos", default=1500, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)

    # params for EXT Transformer layer
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-min_length", default=0, type=int)
    parser.add_argument("-max_length", default=1500, type=int)
    parser.add_argument("-max_tgt_len", default=500, type=int)

    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=5e-6, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_steps", default=None, type=int)
    parser.add_argument("-warmup_steps", default=1, type=int)
    parser.add_argument("-max_grad_norm", default=100, type=float)
    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-save_all_steps", default=False, type=str2bool)
    parser.add_argument("-train_steps", default=20, type=int)
    parser.add_argument("-max_no_train_steps", default=None, type=int)
    #Gradient accumulation
    parser.add_argument("-accum", default=1, type=int)
    
    # PaGAN Generator parameters
    ########################################################################################################################################################################################################
    parser.add_argument('-gan_mode', default=True, type=str2bool)
    parser.add_argument('-generator', default='LSTM', type=str, choices=['LSTM_sent','FC_sent','Transformer_sent','LSTM_doc','Transformer_doc'])
    parser.add_argument('-document_level_generator', default=True, type=str2bool)
    parser.add_argument('-coeff_gen_losses', nargs='*', type=float, default=[1,1,1,1/5,5])
    parser.add_argument('-mean', default=0, type=float)
    parser.add_argument('-std', default=1, type=float)
    parser.add_argument('-g_learning_rate', default=1e-5, type=float)
    parser.add_argument('-use_output', default=False, type=str2bool)
    parser.add_argument('-random_input_size', default=768, type=int)
    parser.add_argument('-size_hidden_layers', default=768, type=int)
    parser.add_argument('-size_embedding', default=768, type=int)
    parser.add_argument('-num_layers', default=2, type=int)
    
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-log_file', default=None)
    parser.add_argument('-log_dir', default='../logs')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument('-n_cpus', default=4, type=int)
    args = parser.parse_args()

    init_logger(args.log_file)
    if args.visible_gpus != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus 
    if args.parallel_computing:
        args.visible_gpus='0'
    device_id=-1 if args.visible_gpus == '-1' else [int(id_gpu) for id_gpu in args.visible_gpus.split(',')]

    if args.mode == 'train':
        if args.baseline=='SummaTRIZ':
            args.doc_classifier='Probabilistic'
        elif args.baseline=='baseline':
            args.doc_classifier='FC'
        if args.load_model is None and args.transfer_learning:
            args.load_model='../models/model_cnn_pretrained.pt'
        else:
            args.transfer_learning=False
    else:
        args.transfer_learning=False
        
    if args.doc_classifier=='Probabilistic':
        #no doc loss for probabilitic model
        args.coeff_gen_losses[1]=0
        args.coeff_model_losses[0]=0
        args.coeff_model_losses[1]=0
        args.coeff_model_losses[4]=0

    if args.need_preprocessing:
        Preprocesseur=Preprocessor(args)
        Preprocesseur.tokenize()
        Preprocesseur.format_to_lines()
        Preprocesseur.format_to_nn()
        del Preprocesseur
        gc.collect()

    if args.mode == 'train' and not args.only_preprocessing:
        train_ext(args, device_id)
    elif args.mode == 'test':
        checkpoint = args.load_model

        # Print or save summaries and probas for test mode
        logging.info("Processing files...")
        with open('../results/patents_analysis.csv', mode='w') as file:
            writer = csv.writer(file, delimiter='?', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Ref patent', 
                'Domain', 
                'Probability of contradiction', 
                'First part Contradiction', 
                'Second part Contradiction', 
                'Context'])
            for ref_patents, summaries,output_probas,prediction_contradiction, str_context in test_ext(args, device_id, checkpoint, 0):
                for ref_patent, summary, output_proba, prediction_c, context in zip(ref_patents, 
                                                                           summaries,
                                                                           output_probas,
                                                                           prediction_contradiction,
                                                                           str_context):

                    # print(summary)
                    # Get patents domain if possible
                    try:
                        with open('../data_patents/input_data/test_directory/'+ref_patent+'/'+ref_patent+'.DOMAIN', mode='r') as domain_f:
                            name_domain = domain_f.read()
                    except:
                        name_domain = 'unknown'

                    # Write results
                    writer.writerow([ref_patent, name_domain, prediction_c, ' '.join(summary['first part']), ' '.join(summary['second part']), str_context])

logging.info("Process finished.")


        

