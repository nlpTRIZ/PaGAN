# -*- coding: utf-8 -*-

import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

import torch
from multiprocessing import Pool
from preprocessing.tokenization import BertTokenizer

from others.logging import logger
from others.utils import clean
from preprocessing.utils import _get_word_ngrams


length=[]

def format_to_bert_(params):
    json_file, args, save_file = params
    tgt2 = None
    
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    
    if args.mode == 'test':
        for d in jobs:
            source, ref_patent= d['src'], d['ref_patent']
            source = [' '.join(s).lower().split() for s in source]
            
            b_data = bert.preprocess(source, None, None, None, None)
            if (b_data is None):
                print(source)
                continue

            src_subtoken_idxs,segments_ids, cls_ids, src_txt = b_data
            b_data_dict = {"src": src_subtoken_idxs, "segs": segments_ids, 'clss': cls_ids,'src_txt': src_txt, "ref_patent": ref_patent}
            datasets.append(b_data_dict)

    else:
        for d in jobs:
            source, tgt, tgt2, ref_patent = d['src'], d['tgt'], d['tgt2'], d['ref_patent']
            if (args.oracle_mode == 'greedy'):
                oracle_ids = greedy_selection(source, tgt)
            elif (args.oracle_mode == 'combination'):
                oracle_ids = combination_selection(source, tgt)

            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]

            
            if (args.oracle_mode == 'greedy'):
                oracle_ids2 = greedy_selection(source, tgt2)
            elif (args.oracle_mode == 'combination'):
                oracle_ids2 = combination_selection(source, tgt2)

            tgt2 = [' '.join(s).lower().split() for s in tgt2]

            # To verify the choice of sentence for golden summary
            ######################################################
            # print("Source",source)
            # print("Target",tgt)
            # for u in oracle_ids:
            #     print(source[u])
            #time.sleep(5)
            ######################################################
            
            b_data = bert.preprocess(source, tgt, tgt2, oracle_ids, oracle_ids2)
            if (b_data is None):
                print(source)
                print(tgt)
                print(tgt2)
                continue

            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, sent_labels_2, tgt_subtoken_idxs2, segments_ids, cls_ids, src_txt, tgt_txt, tgt_txt2 = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "tgt2": tgt_subtoken_idxs2,
                           "src_sent_labels": sent_labels, "src_sent_labels_2": sent_labels_2, "segs": segments_ids, 'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt, "tgt_txt2": tgt_txt2, "ref_patent": ref_patent}
        

            datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count
    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list):

    summary_size = len(abstract_sent_list)

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size=None):

    if summary_size is None:
        summary_size = len(abstract_sent_list)

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge


    return sorted(selected)




class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, tgt2, sent_labels, sent_labels_2, use_bert_basic_tokenizer=False, is_test=False):

        length_source= len(src)
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        if len(src)>self.args.max_src_nsents: src = src[0:len(src)-self.args.max_src_nsents+1]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None
        
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        src_txt = [original_src_txt[i] for i in idxs]
        
        if not self.args.mode == 'test':
            _sent_labels = [0] * length_source
            
            for l in sent_labels:
                _sent_labels[l] = 1

            sent_labels = [_sent_labels[i] for i in idxs]
            sent_labels = sent_labels[:self.args.max_src_nsents]
            sent_labels = sent_labels[:len(cls_ids)]

            tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
            if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
                return None

            tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
            tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
            _sent_labels = [0] * length_source

            for l in sent_labels_2:
                _sent_labels[l] = 1
            sent_labels_2 = [_sent_labels[i] for i in idxs]
            sent_labels_2 = sent_labels_2[:self.args.max_src_nsents]
            sent_labels_2 = sent_labels_2[:len(cls_ids)]

            tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt2]) + ' [unused1]'
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
            
            if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
                return None

            tgt_subtoken_idxs2 = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
            tgt_txt2 = '<q>'.join([' '.join(tt) for tt in tgt2])
            return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, sent_labels_2, tgt_subtoken_idxs2, segments_ids, cls_ids, src_txt, tgt_txt, tgt_txt2

        else:
            return src_subtoken_idxs,segments_ids, cls_ids, src_txt


       

