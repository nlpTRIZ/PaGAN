#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Ext_summarizer,Generator
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def test_ext(args, device_id, pt, step):
    if device_id == -1:
        device = "cpu"
    else:
        device = "cuda"
    logger.info('Device ID %s' %','.join(map(str, device_id)))
    logger.info('Device %s' % device)
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.load_model
    logger.info('Loading model_checkpoint from %s' % test_from)
    model_checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    args.doc_classifier=model_checkpoint['opt'].doc_classifier
    args.nbr_class_neurons=model_checkpoint['opt'].nbr_class_neurons
    model = Ext_summarizer(args, device, model_checkpoint)

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False)
    trainer = trainer = build_trainer(args, device_id, model, None, None, None)

    for ref_patents, summaries,output_probas,prediction_contradiction, str_context in trainer.test(test_iter):
        yield ref_patents, summaries,output_probas,prediction_contradiction, str_context

def train_ext(args, device_id):
    init_logger(args.log_file)
    if device_id == -1:
        device = "cpu"
    else:
        device = "cuda"
    logger.info('Device ID %s' %','.join(map(str, device_id)))
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if device_id != -1:
        torch.cuda.set_device(device_id[0])
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load checkpoint if necessary
    if args.load_model is not None:
        logger.info('Loading model_checkpoint from %s' % args.load_model)
        model_checkpoint = torch.load(args.load_model,
                                map_location=lambda storage, loc: storage)
        if not args.transfer_learning:
            args.doc_classifier=model_checkpoint['opt'].doc_classifier
            args.nbr_class_neurons=model_checkpoint['opt'].nbr_class_neurons
    else:
        model_checkpoint = None
    
    if args.gan_mode and args.load_generator is not None:
        logger.info('Loading generator_checkpoint from %s' % args.load_generator)
        generator_checkpoint = torch.load(args.load_generator,
                                map_location=lambda storage, loc: storage)
        args.generator=generator_checkpoint['opt'].generator
    else:
        generator_checkpoint=None

    # Data generator for training
    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True)
    # Data generator for validation
    def valid_iter_fct(): 
        return data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),args.test_batch_size, device,
                                      shuffle=False)
    
    # Creation model
    model = Ext_summarizer(args, device, model_checkpoint)
    optim = model_builder.build_optim(args, model, model_checkpoint)
    logger.info(model)

    if args.gan_mode:
        # Creation generator if gan
        generator = Generator(args, model.length_embeddings, device, generator_checkpoint)
        optim_generator = model_builder.build_optim_generator(args,generator,generator_checkpoint)
        logger.info(generator)
    else:
        generator = None
        optim_generator = None
    
    trainer = build_trainer(args, device_id, model, generator, optim, optim_generator)
    trainer.train(train_iter_fct, args.train_steps, valid_iter_fct)
