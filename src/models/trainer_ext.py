import os
import time
import random
import json
import glob
import shutil

import numpy as np
import torch
import torch.nn as nn

from models.reporter_ext import ReportMgrBase
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, generator, optim, optim_generator):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argfument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    tensorboard_log_dir=args.log_dir
    report_manager = ReportMgrBase(tensorboard_log_dir)

    trainer = Trainer(args, model, generator, optim, optim_generator, report_manager)

    if (model) and (generator):
        n_params = _tally_parameters(model)+_tally_parameters(generator)
    elif (model):
        n_params = _tally_parameters(model)
    
    logger.info('* number of parameters: %d' % n_params)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
    """

    def __init__(self, args, model, generator, optim, optim_generator,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.generator = generator
        self.optim = optim
        self.optim_generator = optim_generator
        self.report_manager = report_manager
        self.init_loss = 100
        self.no_max_steps = 0
        self.device = "cpu" if self.args.visible_gpus == '-1' else "cuda:"+str(self.args.visible_gpus[0])
        self.weights_loss=torch.tensor(self.args.weights_separate_loss).to(self.device)
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.last_checkpoint = None
        if self.args.doc_classifier == 'Transformer':
            self.contradiction_memory = torch.zeros(10,self.args.size_embedding).to(self.device)
        else:
            self.contradiction_memory = torch.zeros(10,self.args.size_embedding*2).to(self.device)

        #Parallel mode
        if self.args.parallel_computing and torch.cuda.device_count() > 1:
            logger.info("%d GPUs will be used." %torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model,device_ids=[num_gpu for num_gpu in range(torch.cuda.device_count())])
            self.generator = torch.nn.DataParallel(self.generator,device_ids=[num_gpu for num_gpu in range(torch.cuda.device_count())])
        
        # Set model in training mode.
        if (model):
            self.model.train()
        if (generator):
            self.generator.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        self.step = 0
        self.last_result = 100
        self.accum = self.args.accum
        train_iter = train_iter_fct()
        valid_iter = valid_iter_fct()

        #init validation score
        self.validate(valid_iter, self.step)
        valid_iter = valid_iter_fct()
        self.step +=1
        self.model.train()

        self.model.zero_grad()
        if self.generator:
            self.generator.zero_grad()
        self.accumulation_steps = 0

        while self.step <= train_steps:

            for i, batch in enumerate(train_iter):

                self.accumulation_steps+=1
                self._gradient_accumulation(i,
                    batch)

                # Intermediate results
                if self.args.gan_mode and self.step>1 and self.args.evaluate_x_steps!=-1 and i>0 and i%self.args.evaluate_x_steps==0:
                    logger.info("Valid results")
                    self.validate(valid_iter, self.step)
                    valid_iter = valid_iter_fct()
                    self.model.train()

            logger.info("Epoch %d finished." %self.step)
            self.validate(valid_iter, self.step)
            valid_iter = valid_iter_fct()
            self.model.train()

            self.step += 1
            train_iter = train_iter_fct()

        self.end_training()


    def compute_results_statistics(self,labels,score, values):
        """
        Compute FP, FN, TP, TN through all elements of a batch

        Args:
            labels(tensor): labels associated to the batch
            score(tensor): output of the network for the batch
            values(list): all statistics computed for the other batches that will be updated

        Return:
            values(list): updated statistics
        """

        true_positives,true_positives_margin,total_positives,total_positives_margin,tp,fp,tn,fn = values

        for i,(label,scores) in enumerate(zip(labels,score)):
            if label.shape[0] > 1:
                pos_label_first = np.where(label[-2]==1)
                pos_label_second = np.where(label[-1]==1)
            else:
                pos_label_first = np.where(label[0]==1)
                pos_label_second = np.where(label[0]==2)
    
            # Computation true positive
            ##########################################################################################################
            if self.args.classification == 'multi_class' or self.args.classification == 'both':
                pos_score_first = np.argsort(scores[-2,:])
                pos_score_second = np.argsort(scores[-1,:])
                liste = [(pos_score_first,pos_label_first),(pos_score_second,pos_label_second)]
            else:
                pos_score_first = np.argsort(scores[0,:])
                liste = [(pos_score_first,pos_label_first)]
                
            for p,(pos_score,pos_label) in enumerate(liste):
                for pos in range(len(pos_label[0])):
                    
                    if (pos_score[-(pos+1)] in pos_label[0]):
                        true_positives[p]+=1

                if label.shape[0] > 1:
                    enough_length = len(label[p-2,:])>=(np.sum(label[p-2,:]==1)+2)
                else:
                    enough_length = len(label[0])>=(np.sum(label[0]==p+1)+2)
                if enough_length:
                    for pos in range(len(pos_label[0])+2):
                        if (pos_score[-(pos+1)] in pos_label[0]):
                            true_positives_margin[p]+=1
                else:
                    total_positives_margin[p]-=len(pos_label_first[0])

            total_positives[0]+=len(pos_label_first[0])
            total_positives_margin[0]+=len(pos_label_first[0])

            if self.args.classification == 'multi_class' or self.args.classification == 'both':
                total_positives[1]+=len(pos_label_second[0])
                total_positives_margin[1]+=len(pos_label_second[0])

            # Statistics with threshold for classification
            ##########################################################################################################
            probability_threshold = [0.5,0.5]
            if self.args.classification == 'multi_class' or self.args.classification == 'both':
                score_first = (scores[-2,:] > probability_threshold[-2]).astype(np.long)
                score_second = (scores[-1,:] > probability_threshold[-1]).astype(np.long)
                if self.args.classification == 'multi_class':
                    liste = [(label[-2,:],score_first),(label[-1,:],score_second)]
                else:
                    liste = [((label[0,:]==1).astype(np.float),score_first),((label[0,:]==2).astype(np.float),score_second)]
            else:
                score_first = (scores[0,:] > probability_threshold[0]).astype(np.long)
                liste = [(label,score_first)]

            for p,(l,score) in enumerate(liste):
                tp[p] += np.sum(((score==1)*(l==1))==1)
                fp[p] += np.sum(((score==1)*(l==0))==1)
                tn[p] += np.sum(((score==0)*(l==0))==1)
                fn[p] += np.sum(((score==0)*(l==1))==1)

        return (true_positives,true_positives_margin,total_positives,total_positives_margin,tp,fp,tn,fn)


    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        true_positives = [0,0]
        true_positives_margin = [0,0]
        total_positives = [0,0]
        total_positives_margin = [0,0]

        tp = [0,0]
        fp = [0,0]
        tn = [0,0]
        fn = [0,0]

        values = [(true_positives.copy(),true_positives_margin.copy(),total_positives.copy(),total_positives_margin.copy(),tp.copy(),fp.copy(),tn.copy(),fn.copy()) for _ in range(2)]
        self.real_result = [0,0]
        self.fake_result = [0,0]
        self.contradiction_found=[0,0]
        self.G_prediction_for_G_extraction=[0,0]
        self.G_prediction_for_W_extraction=[0,0]
        self.G_prediction_for_no_possible_extraction=[0,0]
        # self.G_prediction_for_possible_extraction=[0,0]
        self.number_co=0

        with torch.no_grad():

            self.D_sent_supervised_loss = 0
            self.D_sent_unsupervised_real_loss = torch.zeros(1).to(self.device)
            self.D_sent_unsupervised_fakes_loss = 0
            self.D_sent_all_loss=[0,0]

            self.D_co_supervised_loss = 0
            self.D_co_unsupervised_real_loss = torch.zeros(1).to(self.device)
            self.D_co_unsupervised_fakes_loss = 0

            self.G_sent_loss = 0
            self.G_doc_loss = 0
            self.G_distance_loss=0
            self.G_feat_match_loss_mean=0
            self.G_feat_match_loss_std=0

            normalization = 0
            normalization_unlabeled = 0
            normalization_fake = 0
            distance_num = 0
            bert_sents=[]
            mask_tot=[]

            for b,batch in enumerate(valid_iter):
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                src_str = batch.src_str
                tgt1 = batch.tgt_str
                tgt2 = batch.tgt_str2
                ref = batch.ref_patent
                contradiction_labels = batch.is_contradiction

                sent_scores, sent_probs, contradiction_scores, contradiction_probs, new_co_labels, best_contradictions, best_probs, concat_embeddings, bert_sents, contradiction_embeddings, self.sent_embeddings_real_data, self.sent_scores_real_data = self.model(src, segs, clss, mask, mask_cls, labels=labels.transpose(0,1), co_labels=contradiction_labels, inference=True)
                
                self.number_co+=len(contradiction_labels[torch.where(contradiction_labels<3)])
                idx=contradiction_scores.size(0)
                if self.args.use_output:
                    self.contradiction_memory = torch.cat((self.contradiction_memory[idx:],contradiction_embeddings[:10]))
                concat_embeddings = concat_embeddings*mask_cls.unsqueeze(-1)
                size_embeddings=list(concat_embeddings.size())[-1]
                self.real_features = concat_embeddings.view(-1,size_embeddings)
                if b==0:
                    self.full_real_features = self.real_features[torch.where(self.real_features.sum(dim=-1)!=0)[0]]
                else:
                    self.full_real_features = torch.cat((self.full_real_features,self.real_features[torch.where(self.real_features.sum(dim=-1)!=0)[0]]),dim=0)
                # Compute statistics
                # Number of correct contradiction prediction on labeled documents
                nbr_labels=contradiction_probs[torch.where(contradiction_labels<2)[0]].size(0)
                if nbr_labels>0:
                    if best_contradictions is not None:
                        best_contradictions=best_contradictions[torch.where(contradiction_labels<2)[0]].transpose(-2,-1)
                        best_probs=best_probs[torch.where(contradiction_labels<2)[0]]
                        if len(best_probs)>1:
                            best_probs=best_probs.squeeze()
                        new_labels = labels.squeeze()[:,torch.where(contradiction_labels<2)[0]].transpose(0,1)
                        # print(new_labels.size())
                        # print(new_labels.sum(dim=-1)[torch.where(contradiction_labels[torch.where(contradiction_labels<2)[0]]==1)[0]][:,1])
                        for i,label in enumerate(new_labels):
                            if label.sum(dim=-1)[0]>0 and label.sum(dim=-1)[1]==0:
                                new_labels[i,1]=new_labels[i,0]
                        # Use only best contradiction
                        if best_contradictions.size(-1)!=1:
                            best_contradictions=best_contradictions[...,0].unsqueeze(-1)
                        one_hot_tensor = torch.zeros_like(new_labels)
                        one_hot_tensor = one_hot_tensor.scatter_(-1, best_contradictions, torch.ones_like(best_contradictions))
                        #look if the chosen sentences are contradictions
                        contradiction_results = (one_hot_tensor*new_labels).sum(dim=(1,2))[torch.where(contradiction_labels[torch.where(contradiction_labels<2)[0]]==1)[0]]
                        self.contradiction_found[1]+=len(contradiction_results)
                        self.contradiction_found[0]+=(contradiction_results==2).sum().cpu().data.numpy()
                        #Use only best score and see if prediction on contradiction is True
                        probs_co_CO=best_probs[torch.where(contradiction_labels[torch.where(contradiction_labels<2)[0]]==1)[0]][torch.where(contradiction_results==2)]#probas when contradiction found
                        probs_no_co_CO=best_probs[torch.where(contradiction_labels[torch.where(contradiction_labels<2)[0]]==1)[0]][torch.where(contradiction_results!=2)]#probas when contradiction not found
                        self.G_prediction_for_G_extraction[1]+=len(probs_co_CO)
                        self.G_prediction_for_G_extraction[0]+=torch.sum(probs_co_CO>0.5).cpu().numpy()
                        self.G_prediction_for_W_extraction[1]+=len(probs_no_co_CO)
                        self.G_prediction_for_W_extraction[0]+=torch.sum(probs_no_co_CO<0.5).cpu().numpy()
                        probs_no_co_no_CO=best_probs[torch.where(contradiction_labels[torch.where(contradiction_labels<2)[0]]==0)[0]]#probas when no contradiction to extract
                        self.G_prediction_for_no_possible_extraction[1]+=len(probs_no_co_no_CO)
                        self.G_prediction_for_no_possible_extraction[0]+=torch.sum(probs_no_co_no_CO<0.5).cpu().numpy()
                
                loss = torch.zeros([2]).to(self.device)
                unsup_loss = torch.zeros([2]).to(self.device)
                fake_loss = torch.zeros([2]).to(self.device)
                
                # Sentence classification loss: Loop over all classifiers
                nbr_contradictions=list(contradiction_labels[torch.where(contradiction_labels==1)[0]].size())[0]
                nbr_unlabeled_data=list(contradiction_labels[torch.where(contradiction_labels==3)[0]].size())[0]
                for i,(value, label, score, proba) in enumerate(zip(values,labels,sent_scores, sent_probs)):
                    if nbr_contradictions>0:
                        label = label[torch.where(contradiction_labels==1)[0]]
                        score_clss = score[torch.where(contradiction_labels==1)[0]][:,:,:self.args.nbr_class_neurons]
                        proba_clss = proba[torch.where(contradiction_labels==1)[0]][:,:,:self.args.nbr_class_neurons]
                        if self.args.nbr_class_neurons==1:
                            mask_c=mask_cls[torch.where(contradiction_labels==1)[0]]
                            score_clss = torch.sigmoid(score_clss)
                            loss_init = -torch.log(1-torch.abs(label[...,0]-score_clss)+1e-7)*mask_c
                            proba_cpu = proba_clss.unsqueeze(-1).transpose(2,1).cpu().data.numpy()
                        else:
                            mask_c=mask_cls[torch.where(contradiction_labels==1)[0]].unsqueeze(-1)
                            score_clss = torch.nn.functional.log_softmax(score_clss,dim=-1)*mask_c
                            one_hot_labels = torch.cat((1-label.float(), label.float()),dim=-1)
                            loss_init = -(score_clss*one_hot_labels)
                            proba_cpu = proba_clss[:,:,1].unsqueeze(-1).transpose(2,1).cpu().data.numpy()
                        loss[i] += loss_init.sum()
                        label_cpu = label.transpose(2,1).cpu().data.numpy()
                        values[i] = self.compute_results_statistics(label_cpu,proba_cpu, value)
                        normalization += mask_c.sum()
                    else:
                        loss = torch.zeros([2]).to(self.device)
                    if self.args.gan_mode and nbr_unlabeled_data>0:
                        score_unlabeled_data = proba[torch.where(contradiction_labels==3)[0]][:,:,-1]
                        unsup_loss[i] += -1 * torch.log(1-score_unlabeled_data+1e-7).sum()
                        normalization_unlabeled += mask_cls[torch.where(contradiction_labels==3)[0]].sum()

                self.D_sent_supervised_loss+=torch.mean(self.weights_loss*loss)
                self.D_sent_all_loss = [loss_tot_i+loss_si for loss_tot_i,loss_si in zip(self.D_sent_all_loss,loss)]
                if self.args.gan_mode and torch.mean(unsup_loss)>0:
                    self.D_sent_unsupervised_real_loss+=torch.mean(unsup_loss)
                
                # Document classification loss
                if self.args.nbr_class_neurons==1:
                    score_classes = torch.sigmoid(contradiction_scores[torch.where(contradiction_labels<=1)[0]][:,0])
                else:
                    score_classes = torch.nn.functional.log_softmax(contradiction_scores[torch.where(contradiction_labels<=1)[0]][:,:2],dim=-1)
                if score_classes.size(0)>0:
                    if self.args.nbr_class_neurons==1:
                        contradiction_labels_clss = contradiction_labels[torch.where(contradiction_labels<=1)[0]]
                        self.D_co_supervised_loss += -torch.log(1-torch.abs(contradiction_labels_clss-score_classes)+1e-7).sum()
                    else:
                        contradiction_labels_clss = contradiction_labels[torch.where(contradiction_labels<=1)[0]].unsqueeze(-1)
                        one_hot_labels_contradiction = torch.cat((1-contradiction_labels_clss.float(), contradiction_labels_clss.float()),dim=1)
                        self.D_co_supervised_loss += -(score_classes*one_hot_labels_contradiction).sum()

                if self.args.gan_mode:
                    # Document unsupervised real loss
                    score_unlabeled_data = contradiction_probs[torch.where(contradiction_labels==3)[0]]
                    if score_unlabeled_data.size(0)>0:
                        score_unlabeled_data = score_unlabeled_data[:,-1]
                        self.D_co_unsupervised_real_loss += -1 * torch.log(1-score_unlabeled_data+1e-7).sum()
                        self.real_result[0]+=torch.sum(score_unlabeled_data<0.5).cpu().data.numpy().tolist()
                        self.real_result[1]+=len(score_unlabeled_data)

                    if b==0 and contradiction_probs[torch.where(contradiction_labels<2)[0]].size(0)>0:
                        take_labeled = True
                    else:
                        take_labeled = False
                    if take_labeled:
                        nbr_generations = contradiction_probs[torch.where(contradiction_labels<2)[0]].size(0)
                    else:
                        nbr_generations = contradiction_probs[torch.where(contradiction_labels==3)[0]].size(0)
                    if nbr_generations>0:
                        if self.args.generator[-4:]=='sent':
                            nbr_generation = nbr_generations
                            nbr_embeddings=max(3,abs(int(random.gauss(8, 2))))
                            random_input = torch.empty((nbr_generation,nbr_embeddings,self.args.random_input_size)).normal_(mean=self.args.mean,std=self.args.std).to(self.device)
                            mask = random_input.sum(dim=-1)!=0
                            fake_embeddings = self.generator(random_input, mask)
                        else:
                            nbr_generation=nbr_generations
                            if self.args.use_output:
                                nbr_generation = min(nbr_generation,len(torch.where(self.contradiction_memory.sum(dim=-1)!=0)[0]))
                                input_mem = self.contradiction_memory[-nbr_generation:]
                                if nbr_generation>1:
                                    random_input = torch.normal(mean=torch.zeros_like(input_mem),std=torch.std(input_mem,axis=0)).to(self.device)
                                else:
                                    random_input = torch.normal(mean=torch.zeros_like(input_mem),std=torch.ones_like(input_mem)).to(self.device)
                                random_input = random_input+input_mem
                            else:
                                random_input = torch.empty((nbr_generation,self.args.random_input_size)).normal_(mean=self.args.mean,std=self.args.std).to(self.device)
                            # print("random_input",random_input.size())
                            fake_embeddings = self.generator(random_input)


                        self.fake_features = fake_embeddings.view(-1,size_embeddings)
                        if b==0:
                            self.full_fake_features = self.fake_features[torch.where(self.fake_features.sum(dim=-1)!=0)[0]]
                        else:
                            self.full_fake_features = torch.cat((self.full_fake_features,self.fake_features[torch.where(self.fake_features.sum(dim=-1)!=0)[0]]),dim=0)
            
                        score_sent_fakes, prob_sent_fakes, score_fakes, prob_fakes, _, _ , _, _, self.sent_embeddings_fake_data, self.sent_scores_fake_data= self.model(None, None, None, None, mask_cls, fake_embeddings, True, self.args.generator)
                        
                        sizes = list(fake_embeddings.size())
                        normalization_fake += sizes[0]*sizes[1]
                        self.fake_result[1]+=len(score_fakes)

                        # Fake documents classification loss
                        prob_fakes = prob_fakes[:,-1]
                        self.G_doc_loss += -1 * torch.log(1.+1e-7-prob_fakes).sum()
                        self.fake_result[0]+=torch.sum(prob_fakes>0.5).cpu().data.numpy().tolist()
                        self.D_co_unsupervised_fakes_loss += -1 * torch.log(prob_fakes+1e-7).sum() 

                        # Fake sentences classification loss
                        for score_sent_fake in prob_sent_fakes:
                            score_sent_fake = score_sent_fake[:,:,-1]
                            self.G_sent_loss += -1 * torch.log(1+1e-7-score_sent_fake).sum()/len(prob_sent_fakes)
                            self.D_sent_unsupervised_fakes_loss += -1 * torch.log(score_sent_fake+1e-7).sum()/len(prob_sent_fakes)

                        if len(fake_embeddings)>1:
                            distance_num += 1
                            # distance loss
                            #######################################""
                            indices=torch.arange(len(fake_embeddings))
                            distance=0
                            for dim in range(len(fake_embeddings)-1):
                                distance += -torch.log(1+1e-7-torch.abs(self.cos_sim(fake_embeddings.view(len(fake_embeddings),-1)[dim],fake_embeddings.view(len(fake_embeddings),-1)[torch.where(indices>dim)]))).sum()
                            self.G_distance_loss+=distance/torch.sum(indices)
                            #######################################

            self.D_sent_supervised_loss = (self.D_sent_supervised_loss/normalization)*self.args.coeff_model_losses[2]
            self.D_sent_supervised_loss = self.D_sent_supervised_loss.cpu().data.numpy()
            # print("D_sent_supervised_loss", self.D_sent_supervised_loss)
            self.D_sent_all_loss = torch.Tensor(self.D_sent_all_loss).cpu()/normalization.cpu()*self.args.coeff_model_losses[2]
            self.D_sent_all_loss = self.D_sent_all_loss.data.numpy()
            
            self.D_co_supervised_loss = (self.D_co_supervised_loss).sum()/self.number_co*self.args.coeff_model_losses[1]
            self.D_co_supervised_loss = self.D_co_supervised_loss.cpu().data.numpy()
            # print("D_co_supervised_loss", self.D_co_supervised_loss)

            if self.args.gan_mode:
                self.G_feat_match_loss_mean = torch.norm(self.full_real_features.detach().mean(dim=0)-self.full_fake_features.mean(dim=0), p=None,dim=-1)/2
                self.G_feat_match_loss_mean.cpu().data.numpy()
                self.G_feat_match_loss_std = torch.norm(torch.std(self.real_features.detach(),dim=0)-torch.std(self.full_fake_features,dim=0))/2
                self.G_feat_match_loss_std.cpu().data.numpy()

                self.G_distance_loss/= distance_num

                self.D_sent_unsupervised_fakes_loss = self.D_sent_unsupervised_fakes_loss/normalization_fake*self.args.coeff_model_losses[5]
                self.D_sent_unsupervised_fakes_loss = self.D_sent_unsupervised_fakes_loss.cpu().data.numpy()
                # print("D_sent_unsupervised_fakes_loss", self.D_sent_unsupervised_fakes_loss)

                if normalization_unlabeled>0:
                    self.D_sent_unsupervised_real_loss = self.D_sent_unsupervised_real_loss.sum()/normalization_unlabeled*self.args.coeff_model_losses[3]
                self.D_sent_unsupervised_real_loss = self.D_sent_unsupervised_real_loss.cpu().data.numpy()
                # print("D_sent_unsupervised_real_loss", self.D_sent_unsupervised_real_loss)

                self.D_co_unsupervised_fakes_loss = self.D_co_unsupervised_fakes_loss/self.fake_result[1]*self.args.coeff_model_losses[4]
                self.D_co_unsupervised_fakes_loss = self.D_co_unsupervised_fakes_loss.cpu().data.numpy()
                # print("D_co_unsupervised_fakes_loss", self.D_co_unsupervised_fakes_loss)

                self.D_co_unsupervised_real_loss = self.D_co_unsupervised_real_loss/self.real_result[1]*self.args.coeff_model_losses[0]
                self.D_co_unsupervised_real_loss = self.D_co_unsupervised_real_loss.cpu().data.numpy()
                # print("D_co_unsupervised_real_loss", self.D_co_unsupervised_real_loss)

                self.G_sent_loss= self.G_sent_loss/normalization_fake
                self.G_sent_loss = self.G_sent_loss.cpu().data.numpy()
                # print("G_sent_loss", self.G_sent_loss)

                self.G_doc_loss= self.G_doc_loss/normalization_fake
                self.G_doc_loss = self.G_doc_loss.cpu().data.numpy()
                # print("G_doc_loss", self.G_doc_loss)
                                    
            else:
                self.D_co_unsupervised_fakes_loss = 0
                self.D_co_unsupervised_real_loss = 0
                self.D_sent_unsupervised_real_loss=0
                self.D_sent_unsupervised_fakes_loss=0
                self.G_sent_loss = 0
                self.G_doc_loss = 0

            self.D_total_loss = self.D_sent_supervised_loss+\
                                  self.D_sent_unsupervised_real_loss+\
                                  self.D_sent_unsupervised_fakes_loss+\
                                  self.D_co_supervised_loss+\
                                  self.D_co_unsupervised_fakes_loss+\
                                  self.D_co_unsupervised_real_loss

            self.G_total_loss = self.G_doc_loss+\
                                self.G_sent_loss+\
                                self.G_feat_match_loss_mean+\
                                self.G_feat_match_loss_std+\
                                self.G_distance_loss


            self.D_losses = {'D_total_loss':self.D_total_loss,
                           'D_sent_supervised_loss':self.D_sent_supervised_loss,
                           'D_sent_all_loss':self.D_sent_all_loss,
                           'D_sent_unsupervised_real_loss':self.D_sent_unsupervised_real_loss,
                           'D_sent_unsupervised_fakes_loss':self.D_sent_unsupervised_fakes_loss,
                           'D_co_supervised_loss':self.D_co_supervised_loss,
                           'D_co_unsupervised_real_loss':self.D_co_unsupervised_real_loss,
                           'D_co_unsupervised_fakes_loss':self.D_co_unsupervised_fakes_loss
                           }
            
            self.G_losses = {'G_total_loss':self.G_total_loss,
                           'G_doc_loss':self.G_doc_loss,
                           'G_sent_loss':self.G_sent_loss,
                           'G_feat_match_loss_mean':self.G_feat_match_loss_mean,
                           'G_feat_match_loss_std':self.G_feat_match_loss_std,
                           'G_distance_loss':self.G_distance_loss
                           }
                    
            for i,value in enumerate(values):
                if i==0:
                    (self.true_positives_tot,
                     self.true_positives_margin_tot,
                     self.total_positives_tot,
                     self.total_positives_margin_tot,
                     self.tp_tot,
                     self.fp_tot,
                     self.tn_tot,
                     self.fn_tot) = value
                else:
                    (true_positives,true_positives_margin,total_positives,total_positives_margin,tp,fp,tn,fn) = value
                    self.true_positives_tot[i]=true_positives[0]
                    self.true_positives_margin_tot[i]=true_positives_margin[0]
                    self.total_positives_tot[i]=total_positives[0]
                    self.total_positives_margin_tot[i]=total_positives_margin[0]
                    self.tp_tot[i]=tp[0]
                    self.fp_tot[i]=fp[0]
                    self.tn_tot[i]=tn[0]
                    self.fn_tot[i]=fn[0]

            self.report_step(step)

            self._save(step)

            # with open('test.txt', 'w') as outfile:
            #     json.dump(data, outfile)


    def test(self, test_iter):
        """
        Determine the sentences of the summary with their probability

        Args:
            test_iter(iterator): data generator for test set
            
        Return:
            summaries(dict): dictionnary containing all sentences belonging to summary for all test data
            probabilities(dict): dictionnary containing all summary sentences probabilities
        """

        self.model.eval()

        with torch.no_grad():

            for batch in test_iter:
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                src_txt = batch.src_str
                indic_piece = batch.indic_piece
                len_cut = batch.len_cut
                ref_patent=batch.ref_patent

                # Get network output scores
                _, sent_scores, _, contradiction_scores, _, _, _, _, _ , _ = self.model(src, segs, clss, mask, mask_cls)
                # print("sent_scores",sent_scores[0].size())
                # Standardization of output shapes and values
                probas_tensor_heads = self.standardize_output(sent_scores)
                # print("probas_tensor_heads",probas_tensor_heads.size())
                #RECREATE THE FULL TENSORS FROM PIECES IF PIECES
                result_tensor = self.recreate_output(indic_piece,probas_tensor_heads,src_txt,len_cut)
                # print("result_tensor",result_tensor.size())
                # (parts, batch_size, scores, num_sentences)
                # Compute lengths summaries
                length_summary = self.compute_length_summaries(result_tensor)
                # print("length_summary",length_summary)

                selected_ids = torch.argsort(-result_tensor)
                sorted_tensor, _ = torch.sort(-result_tensor)
                sorted_tensor=-sorted_tensor.squeeze(2)

                if result_tensor.size()[0]>1:
                    list_part = ['first part','second part']
                else:
                    list_part = [self.args.classification+' part']

                # Retrieve the summaries sentences
                # single out length src string
                length_src_str=[]
                for src_stri, indic in zip(batch.src_str,indic_piece):
                    if indic==-1:
                        length_src_str.append(len(src_stri))

                # single out src string
                src_str_final = []
                for elmt in batch.src_str:
                    if elmt not in src_str_final:
                        src_str_final.append(elmt)

                try:
                    # logger.info("New batch")
                    pred_part = []
                    context_part = []
                    for p,part in enumerate(list_part):
                        _pred_elmt = []
                        _context_elmt = []
                        for i in range(result_tensor.size()[1]):
                            # logger.info("New document")
                            length_sum = self.get_length_summary(length_summary,p,i)
                            select_ids = selected_ids.squeeze(2)[p][i][0:length_sum].cpu().data.numpy()
                            _pred = []
                            context = [max(0,min(select_ids)-5),min(length_src_str[i]-1, max(select_ids)+5)]
                            for j in select_ids:
                                if (j >= length_src_str[i]):
                                    continue
                                candidate = src_str_final[i][j].strip()
                                # print(candidate)
                                _pred.append(candidate)
                                if (len(_pred) >= length_sum):
                                    break
                            _pred_elmt.append(_pred)
                            _context_elmt.append(context)
                        pred_part.append(_pred_elmt)
                        context_part.append(_context_elmt)
                except:
                    logger.info("Encountered problem during summary writing.")
                    # print("src_str_final",src_str_final)
                    # print("i",i)
                    # print("j",j)
                    continue

                # Build context for summary
                context_part = np.array(context_part)
                context = np.concatenate((np.amin(context_part,axis=(0,-1), keepdims=True),np.amax(context_part,axis=(0,-1), keepdims=True)),axis=-1)[0].tolist()
                str_context = []
                # src_str_final = np.array(src_str_final)
                for i,elmt in enumerate(context):
                    str_context.append(' '.join(src_str_final[i][elmt[0]:elmt[1]]))


                # All results summaries and probabilities are sent to a dictionnary
                summaries = []
                probabilities = []
                for elt in range(len(pred_part[0])):
                    dict_elt = dict()
                    dict_probas = dict()
                    for p,part in enumerate(list_part):
                        length_sum = self.get_length_summary(length_summary,p,elt)
                        dict_elt[part]=pred_part[p][elt]
                        dict_probas[part]=sorted_tensor[p][elt][0:len(pred_part[p][elt])].cpu().data.numpy().tolist()

                    summaries.append(dict_elt)
                    probabilities.append(dict_probas)

                ## rewrite contradictions scores adn ref_patents
                contradiction_scores_lst = []
                ref_patents_lst = []
                size=1
                total_score=0
                if self.args.nbr_class_neurons==2:
                    scores_document = torch.nn.functional.softmax(contradiction_scores[:,0:2],dim=-1)[:,1].cpu().data.numpy().tolist()
                else:
                    scores_document = torch.sigmoid(contradiction_scores[:,0]).cpu().data.numpy().tolist()
                for i,score_c in enumerate(scores_document):
                    if indic_piece[i]!=-1:
                        total_score += score_c
                        size+=1
                    else:
                        if size==1:
                            total_score=score_c
                        contradiction_scores_lst.append(total_score/size)
                        size = 1
                        total_score=0
                        ref_patents_lst.append(ref_patent[i])


                yield ref_patents_lst, summaries, probabilities, contradiction_scores_lst, str_context



    def standardize_output(self,sent_scores):
        # Standardization of output shapes and values
        # (parts, batch_size, scores(1,0), num_sentences)
        for u,score in enumerate(sent_scores):
            if self.args.nbr_class_neurons==2:
                if u==0:
                    sent_scores_f = torch.nn.functional.softmax(score.unsqueeze(0)[...,:2],dim=-1)[...,1].unsqueeze(-1)
                else:
                    sent_scores_f = torch.cat((sent_scores_f,torch.nn.functional.softmax(score.unsqueeze(0)[...,:2],dim=-1)[...,1].unsqueeze(-1)),dim=0)
            else:
                if u==0:
                    sent_scores_f=torch.sigmoid(score[...,0].unsqueeze(0).unsqueeze(-1))
                else:
                    sent_scores_f = torch.cat((sent_scores_f,torch.sigmoid(score[...,0].unsqueeze(0).unsqueeze(-1))),dim=0)
        probas_tensor_heads = sent_scores_f.permute(0,1,3,2)
        return probas_tensor_heads


    def recreate_output(self,indic_piece,probas_tensor_heads,batch_src_string,len_cut):
        #RECREATE THE FULL TENSORS FROM PIECES IF PIECES
        # Sort the output tensors in a list whether they are pieces of a document or not
        result = []

        for i,indic in enumerate(indic_piece):
            if indic!=-1 and (i==0 or last_indic==-1):
                accum=[probas_tensor_heads[:,i,:,:].unsqueeze(1)]
            elif indic!=-1 and last_indic!=-1:
                accum.append(probas_tensor_heads[:,i,:,:].unsqueeze(1))
            elif indic==-1 and i>0 and last_indic!=-1:
                accum.append(probas_tensor_heads[:,i,:,:].unsqueeze(1))
                result.append(accum)
            else:
                result.append([probas_tensor_heads[:,i,:,:].unsqueeze(1)])
            last_indic=indic

        #compute final last dim of tensor without pieces:
        max_size = max([len(src_str) for src_str in batch_src_string])
        # print(max_size)

        # takes a list of pieces and recreates the full output
        len_cut_final = []
        for elmt, indic in zip(len_cut,indic_piece):
            if indic==-1:
                len_cut_final.append(elmt)

        for i,tensor_list in enumerate(result):
            shape_single_tensor = torch.zeros((probas_tensor_heads.size()[0],
                                    1,
                                    probas_tensor_heads.size()[2],
                                    max_size))
            end_size = 0
            for j,tensor in enumerate(tensor_list):
                size_piece = len_cut_final[i][j]
                # add other way than concatenation to fusion
                shape_single_tensor[:,:,:,end_size:end_size+size_piece]=tensor[:,:,:,0:size_piece]
                end_size += size_piece
            if i==0:
                result_tensor=shape_single_tensor
            else:
                result_tensor = torch.cat((result_tensor,shape_single_tensor),dim=1)
        return result_tensor


    def compute_length_summaries(self,result_tensor):
        # Compute lengths summaries
        if self.args.test_threshold is not None:
            # if result_tensor.size()[0]>1:
            length_summary = torch.sum(result_tensor>self.args.test_threshold,dim=(2,3)).int().cpu().data.numpy().tolist()
        elif self.args.length_summary is not None:
            length_summary = self.args.length_summary
        else:
            length_summary =3
        return length_summary

    def get_length_summary(self,length_summary,part,elmt):
        if isinstance(length_summary, list):
            length_sum = length_summary[part][elmt]
        else:
            length_sum = length_summary
        return length_sum

    def compute_supervised_loss(self, mask_cls, sent_scores, sent_probs, labels,contradiction_labels):
        
        loss = torch.zeros([2]).to(self.device)
        unsup_loss = torch.zeros([2]).to(self.device)
        nbr_contradictions=list(contradiction_labels[torch.where(contradiction_labels==1)[0]].size())[0]
        normalization=0
        normalization_unlabeled = 0
        nbr_contradictions=list(contradiction_labels[torch.where(contradiction_labels==1)[0]].size())[0]
        nbr_unlabeled_data=list(contradiction_labels[torch.where(contradiction_labels==3)[0]].size())[0]
        ind=0;
        for i,(label, score, proba) in enumerate(zip(labels,sent_scores, sent_probs)):
            if nbr_contradictions>0:
                label = label[torch.where(contradiction_labels==1)[0]]
                score_clss = score[torch.where(contradiction_labels==1)[0]][:,:,:self.args.nbr_class_neurons]
                if self.args.nbr_class_neurons==1:
                    score_clss = torch.sigmoid(score_clss)
                else:
                    score_clss = torch.nn.functional.log_softmax(score_clss,dim=-1)
                if self.args.classification == 'separate':
                    if self.args.nbr_class_neurons==1:
                        mask_c=mask_cls[torch.where(contradiction_labels==1)[0]]
                        loss_init = -torch.log(1-torch.abs(label[...,0]-score_clss)+1e-7)*mask_c
                    else:
                        mask_c=mask_cls[torch.where(contradiction_labels==1)[0]].unsqueeze(-1)
                        one_hot_labels = torch.cat((1-label.float(), label.float()),dim=-1)
                        loss_init = -(score_clss*one_hot_labels)
                    loss[i] += loss_init.sum()
                normalization += mask_c.sum()
            else:
                normalization=1
            if nbr_unlabeled_data>0:
                score_unlabeled_data = proba[torch.where(contradiction_labels==3)[0]][:,:,-1]
                mask_u=mask_cls[torch.where(contradiction_labels==3)[0]]
                unsup_loss[i] += -1*(torch.log(1-score_unlabeled_data)*mask_u).sum()
                normalization_unlabeled += mask_u.sum()
            else:
                normalization_unlabeled = 1
        D_sent_supervised_loss=torch.mean(self.weights_loss*loss)/normalization
        D_sent_unsupervised_real_loss=torch.mean(unsup_loss)/normalization_unlabeled
        
        return D_sent_supervised_loss, D_sent_unsupervised_real_loss


    def _gradient_accumulation(self, i, batch):

        # for i,batch in enumerate(true_batchs):

        src = batch.src
        labels = batch.src_sent_labels
        segs = batch.segs
        clss = batch.clss
        mask = batch.mask_src
        mask_cls = batch.mask_cls
        src_str = batch.src_str
        contradiction_labels = batch.is_contradiction

        if len(src)!=self.args.batch_size:
            return

        D_total_real_loss=0
        if self.args.gan_mode:
            # print(src.size())
            sent_scores, sent_probs, contradiction_scores, contradiction_probs, new_co_labels, concat_embeddings, bert_sents, contradiction_embeddings, self.sent_embeddings_real_data, self.sent_scores_real_data = self.model(src, segs, clss, mask, mask_cls, None, labels=labels.transpose(0,1), co_labels=contradiction_labels)

            idx=contradiction_scores.size(0)
            if self.args.use_output:
                self.contradiction_memory = torch.cat((self.contradiction_memory[idx:],contradiction_embeddings[:10]))
            concat_embeddings = concat_embeddings*mask_cls.unsqueeze(-1)
            self.concat_embeddings = concat_embeddings
            size_embeddings=list(concat_embeddings.size())[-1]
            self.real_features = concat_embeddings.view(-1,size_embeddings)
            self.real_features = self.real_features[torch.where(self.real_features.sum(dim=-1)!=0)[0]]

            # Unlabedled loss
            #################################
            score_unlabeled_data = contradiction_probs[torch.where(contradiction_labels==3)[0]]
            new_co_labels=(torch.ones_like(score_unlabeled_data[:,-1])*3).to(score_unlabeled_data.get_device())
            
            if score_unlabeled_data.size(0)>0 and self.args.document_level_generator:
                score_unlabeled_data = score_unlabeled_data[:,-1]
                D_co_unsupervised_real_loss = -1*torch.log(1-score_unlabeled_data).mean()
                if D_co_unsupervised_real_loss>0:
                    D_total_real_loss = D_co_unsupervised_real_loss*self.args.coeff_model_losses[0]
                    # print("D_co_unsupervised_real_loss",D_co_unsupervised_real_loss) 
                else:
                    D_total_real_loss=0
            else:
                D_total_real_loss=0
            #################################
        else:
            sent_scores, sent_probs, contradiction_scores, contradiction_probs, _, _, _, _, _, _ = self.model(src, segs, clss, mask, mask_cls, None, labels=labels.transpose(0,1), co_labels=contradiction_labels)
        
        # Contradiction score
        if self.args.nbr_class_neurons==1:
            score_classes = torch.sigmoid(contradiction_scores[torch.where(contradiction_labels<=1)[0]][:,:self.args.nbr_class_neurons])
        else:
            score_classes = torch.nn.functional.log_softmax(contradiction_scores[torch.where(contradiction_labels<=1)[0]][:,:2],dim=-1)
        # Contradiction classification loss
        #################################
        if score_classes.size(0)>0:
            contradiction_labels_clss = contradiction_labels[torch.where(contradiction_labels<=1)[0]]
            if self.args.nbr_class_neurons==1:
                D_co_supervised_loss =-torch.log(1-torch.abs(contradiction_labels_clss-score_classes)+1e-7).sum()
            else:
                contradiction_labels_clss=contradiction_labels_clss.unsqueeze(-1)
                one_hot_labels_contradiction = torch.cat((1-contradiction_labels_clss.float(), contradiction_labels_clss.float()),dim=1)
                D_co_supervised_loss = -(score_classes*one_hot_labels_contradiction).sum(dim=-1).mean()
            if self.args.gan_mode:
                D_total_real_loss += D_co_supervised_loss*self.args.coeff_model_losses[1]
            else:
                D_total_real_loss = D_co_supervised_loss*self.args.coeff_model_losses[1]

            # print("D_co_supervised_loss",D_co_supervised_loss)
        #################################

        # Sentence classification loss
        #################################
        D_sent_supervised_loss, D_sent_unsupervised_real_loss = self.compute_supervised_loss(mask_cls=mask_cls,
                                                       sent_scores=sent_scores,
                                                       sent_probs =sent_probs, 
                                                       labels=labels,
                                                       contradiction_labels=contradiction_labels)
        if D_sent_supervised_loss > 0:
            D_total_real_loss += D_sent_supervised_loss*self.args.coeff_model_losses[2]
            # print("D_sent_supervised_loss",D_sent_supervised_loss)
        
        
        if self.args.gan_mode and D_sent_unsupervised_real_loss > 0:
            D_total_real_loss += D_sent_unsupervised_real_loss*self.args.coeff_model_losses[3]
            # print("D_sent_unsupervised_real_loss",D_sent_unsupervised_real_loss)
        #################################
        
        if self.args.gan_mode:
            # number fakes = number no fakes
            
            nbr_generation = 8
            if self.args.generator[-4:]=='sent':
                nbr_embeddings=max(3,abs(int(random.gauss(8, 2))))
                    
                random_input = torch.empty((nbr_generation,nbr_embeddings,self.args.random_input_size)).normal_(mean=0,std=self.args.std).to(self.device)
                # print("random",random_input.size())
                mask = random_input.sum(dim=-1)!=0
                fake_embeddings = self.generator(random_input, mask)
            else:
                if self.args.use_output:
                    # print(self.contradiction_memory.size())
                    nbr_generation = min(nbr_generation,len(torch.where(self.contradiction_memory.sum(dim=-1)!=0)[0]))
                    # print("nbr_generation", nbr_generation)
                    input_mem = self.contradiction_memory[-nbr_generation:].detach()
                    if nbr_generation>1:
                        random_input = torch.normal(mean=torch.zeros_like(input_mem),std=torch.std(input_mem,axis=0)/5).to(self.device)
                    else:
                        random_input = torch.normal(mean=torch.zeros_like(input_mem),std=torch.ones_like(input_mem)/5).to(self.device)
                    random_input += input_mem
                else:
                    random_input = torch.empty((nbr_generation,self.args.random_input_size)).normal_(mean=0,std=self.args.std).to(self.device)
                fake_embeddings = self.generator(random_input)
            
            score_sent_fakes, prob_sent_fakes, score_fakes, prob_fakes,_, _,_,_, self.sent_embeddings_fake_data, self.sent_scores_fake_data = self.model(src, segs, clss, mask, mask_cls, fake_embeddings.detach(), True, self.args.generator, labels=None, co_labels=None)
            
            # Fakes loss
            #################################
            
            if self.args.document_level_generator:
                D_co_unsupervised_fakes_loss = -1*torch.log(prob_fakes[:,-1]).mean()
            else:
                D_co_unsupervised_fakes_loss=0

            if D_co_unsupervised_fakes_loss>0:
                D_total_fake_loss = D_co_unsupervised_fakes_loss*self.args.coeff_model_losses[4]
            else:
                D_total_fake_loss=0

            # print("D_co_unsupervised_fakes_loss",D_co_unsupervised_fakes_loss)
            #################################

            for score_sent_fake in prob_sent_fakes:
                score_sent_fake = score_sent_fake[:,:,-1]
                D_sent_unsupervised_fakes_loss = -1*torch.log(score_sent_fake).mean()
                if D_sent_unsupervised_fakes_loss > 1e-4:
                    D_total_fake_loss += D_sent_unsupervised_fakes_loss/len(prob_sent_fakes)*self.args.coeff_model_losses[5]
                # print("D_sent_unsupervised_fakes_loss",D_sent_unsupervised_fakes_loss)

            total_D_loss=D_total_real_loss+D_total_fake_loss
            if isinstance(total_D_loss, torch.Tensor) and not torch.isnan(total_D_loss):
                (total_D_loss/self.accum).backward()
                if self.accumulation_steps%self.accum==0:
                    self.optim.step()
                    self.model.zero_grad()
                    
            
            _, prob_sent_fakes, score_fakes, prob_fakes,_, _, _, contradiction_embeddings, _, self.sent_scores_fake_data = self.model(src, segs, clss, mask, mask_cls, fake_embeddings, True, self.args.generator, labels=None, co_labels=None)
            self.fake_features = fake_embeddings.view(-1,size_embeddings)
            self.fake_features = self.fake_features[torch.where(self.fake_features.sum(dim=-1)!=0)[0]]
            G_feat_match_loss_mean = torch.norm(self.real_features.detach().mean(dim=0)-self.fake_features.mean(dim=0), p=None,dim=-1)
            # print("G_feat_match_loss_mean",G_feat_match_loss_mean)
            G_feat_match_loss_std = torch.norm(torch.std(self.real_features.detach(),dim=0)-torch.std(self.fake_features,dim=0))
            # print("G_feat_match_loss_std",G_feat_match_loss_std)

            if self.args.document_level_generator:
                G_doc_loss = -1 * torch.log(1+1e-7-prob_fakes[:,-1]).mean()
            else:
                G_doc_loss = 0
            # print("G_loss_co",G_loss)
            # sent generator loss
            G_sent_loss=0
            for score_sent_fake in prob_sent_fakes:
                score_sent_fake = score_sent_fake[:,:,1]
                G_sent_loss += -1 * torch.log(1+1e-7-score_sent_fake).mean()/len(prob_sent_fakes)
            # print("G_loss_after_co",G_loss)
            
            #distance loss
            G_distance_loss=0
            # if self.args.generator!='Transformer_sent':
            indices=torch.arange(len(fake_embeddings))
            # print("fake_emb",fake_embeddings)
            for dim in range(len(fake_embeddings)-1):
                cosine_sim = self.cos_sim(fake_embeddings.view(len(fake_embeddings),-1)[dim],fake_embeddings.view(len(fake_embeddings),-1)[torch.where(indices>dim)])
                G_distance_loss += -torch.log(1+1e-7-torch.max(cosine_sim,torch.zeros_like(cosine_sim))).sum()
            G_distance_loss=G_distance_loss/torch.sum(indices)

            total_G_loss = [G_feat_match_loss_mean,G_doc_loss,G_sent_loss,G_feat_match_loss_std,G_distance_loss]
            # print("total_G_loss",total_G_loss)
            total_G_loss = sum([loss*coeff for loss,coeff in zip(total_G_loss,self.args.coeff_gen_losses)])

            if isinstance(total_G_loss, torch.Tensor) and not torch.isnan(total_G_loss):
                (total_G_loss/self.accum).backward()
                if self.accumulation_steps%self.accum==0:
                    self.optim_generator.step()
                    self.generator.zero_grad()
            
        else:
            total_D_loss=D_total_real_loss
            (total_D_loss/self.accum).backward()
            if self.accumulation_steps%self.accum==0:
                self.optim.step()
                self.model.zero_grad()


    def _save(self, step):
        if self.D_sent_supervised_loss+self.D_co_supervised_loss<self.last_result or self.args.save_all_steps:
            self.no_max_steps = 0
            self.last_result = self.D_sent_supervised_loss+self.D_co_supervised_loss

            if not self.args.save_all_steps:
                logger.info("New minimal supervised loss: %.3f" % self.last_result)
                if self.last_checkpoint is not None:
                    for checkpoint in self.last_checkpoint:
                        os.remove(checkpoint)
            self.last_checkpoint = []

            if self.args.gan_mode:
                models=[self.generator,self.model]
                optims=[self.optim_generator,self.optim]
                names=['generator','model']
            else:
                models=[self.model]
                optims=[self.optim]
                names=['model']
            for model,optim,name in zip(models,optims,names):
                model_state_dict = model.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'opt': self.args,
                    'optim': optim,
                }
                if name=='model':
                    checkpoint_path = os.path.join(self.args.model_path, '%s_step_%d_%s_first_%d_second_%d_co_%d.pt' % (name,step,self.args.doc_classifier,self.true_positives_tot[0],self.true_positives_tot[1],self.contradiction_found[0]))
                else:
                    checkpoint_path = os.path.join(self.args.model_path, '%s_step_%d_%s_first_%d_second_%d_co_%d.pt' % (name,step,self.args.generator,self.true_positives_tot[0],self.true_positives_tot[1],self.contradiction_found[0]))
                logger.info("Saving checkpoint %s" % checkpoint_path)
                self.last_checkpoint.append(checkpoint_path)

                torch.save(checkpoint, checkpoint_path)
                self.save_data(checkpoint_path)
        else:
            if self.args.max_no_train_steps:
                self.no_max_steps+=1
                if self.no_max_steps>self.args.max_no_train_steps:
                    self.step=self.args.train_steps+1


    def save_data(self, checkpoint_path):
        for key in self.D_losses.keys():
            if isinstance(self.D_losses[key],np.ndarray) and self.D_losses[key].shape or isinstance(self.D_losses[key],list):
                self.D_losses[key]=[float(elmt) for elmt in self.D_losses[key]]
            else:
                self.D_losses[key]= [float(self.D_losses[key])]
        for key in self.G_losses.keys():
            if isinstance(self.G_losses[key],list):
                self.G_losses[key]=[float(elmt) for elmt in self.G_losses[key]]
            else:
                self.G_losses[key]= [float(self.G_losses[key])]

        output_dict = {'D_losses':self.D_losses,
                       'G_losses':self.G_losses,
                       'Found sentences': [int(elmt) for elmt in self.true_positives_tot],
                       'Found sentences with margin':[int(elmt) for elmt in self.true_positives_margin_tot],
                       'Total sentences':[int(elmt) for elmt in self.total_positives_tot],
                       'TP':[int(elmt) for elmt in self.tp_tot],
                       'FP':[int(elmt) for elmt in self.fp_tot],
                       'TN':[int(elmt) for elmt in self.tn_tot],
                       'FN':[int(elmt) for elmt in self.fn_tot],
                       'Found contradictions': [int(elmt) for elmt in self.contradiction_found],
                       'G_prediction_for_G_extraction': [int(elmt) for elmt in self.G_prediction_for_G_extraction],
                       'G_prediction_for_W_extraction': [int(elmt) for elmt in self.G_prediction_for_W_extraction],
                       'G_prediction_for_no_possible_extraction': [int(elmt) for elmt in self.G_prediction_for_no_possible_extraction],
                       'checkpoint_path': checkpoint_path}

        with open('../results/results_'+ 
                    str(self.args.generator) + '_' + 
                    str(self.args.doc_classifier) + '_' + 
                    str(self.args.real_ratio) + '_' +
                    str(self.args.num_split) +
                    '.txt', 'w') as outfile:
            json.dump(output_dict, outfile)       


    def report_step(self, step):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            self.report_manager.report_step(
                step,
                self.D_losses,
                self.G_losses,
                self.true_positives_tot,
                self.true_positives_margin_tot,
                self.total_positives_tot,
                self.total_positives_margin_tot,
                self.tp_tot,
                self.fp_tot,
                self.tn_tot,
                self.fn_tot,
                self.contradiction_found,
                self.G_prediction_for_G_extraction,
                self.G_prediction_for_W_extraction,
                self.G_prediction_for_no_possible_extraction,
                self.real_result,
                self.fake_result)


    def end_training(self):
        self.report_manager.end_report()

