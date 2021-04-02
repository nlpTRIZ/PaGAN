import copy
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
import numpy as np

from models.modules import ExtTransformerEncoder, doc_classifier, MLP_classifier, c_sampling, c_all
from models.optimizers import Optimizer
from others.logging import logger

def build_optim(args, model, checkpoint=None):
    """ Build optimizer """
    if checkpoint is not None and not args.transfer_learning:
        logger.info('Loading model optimizer...')
        optim = checkpoint['optim']
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)
    optim.set_parameters(list(model.named_parameters()))
    
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    return optim


def build_optim_generator(args, model, checkpoint=None):
    """ Build optimizer """

    if checkpoint is not None and not args.transfer_learning:
        logger.info('Loading generator optimizer...')
        optim = checkpoint['optim']
    else:
        optim = Optimizer(
            args.optim, args.g_learning_rate, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)
    optim.set_parameters(list(model.named_parameters()))

    # optim = torch.optim.Adam(model.parameters(), lr=args.g_learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    return optim

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class Bert_Sum_embedding(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(Bert_Sum_embedding, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if self.args.transfer_learning:
            self.load_model(checkpoint)


    def load_model(self,checkpoint):
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

    def forward(self, src, segs, clss, mask_src, mask_cls, full_through=True):
        if full_through:
            top_vec = self.bert(src, segs, mask_src)
            #clss are the indexes of the '.' vectors which are meant to be the sentences representations
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        else:
            sents_vec = src

        sents_vec = sents_vec * mask_cls[:, :, None].float()
        embeddings = self.ext_layer(sents_vec, mask_cls)
        return embeddings, sents_vec

class Ext_summarizer(nn.Module):
    def __init__(self,args,device,checkpoint):
        super(Ext_summarizer, self).__init__()
        self.args = args
        self.device = device
        self.embedding_model=Bert_Sum_embedding(self.args,self.device,checkpoint)
        self.length_embeddings = 0
        self.length_embeddings+=self.embedding_model.bert.model.config.hidden_size
        self.classifier_sup = []
        for _ in range(2):
            self.classifier_sup.append(MLP_classifier(self.args,self.length_embeddings))
        self.classifier_sup = nn.ModuleList(self.classifier_sup)

        self.doc_classifier = doc_classifier(self.length_embeddings, n_hid_lstm=768, num_layers=1, device=self.device, use_packed_data=self.args.use_packed_data, doc_classifier=self.args.doc_classifier, args=self.args)

        if not self.args.transfer_learning:
            self.load_model(checkpoint)

        self.to(device)

    def load_model(self,checkpoint):
        if checkpoint is not None and not self.args.transfer_learning:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k[0:6]=='module':
                    name = k[7:] # remove module.
                else:
                    name=k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=True)
        elif checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

    def forward(self, src, segs, clss, mask_src, mask_cls, fake_representations=None, fake_only=False, generator=None, labels=None, co_labels=None, inference=False):
        
        if fake_only:
            mask_cls = fake_representations.sum(dim=-1)!=0
            concat_embeddings=fake_representations
            bert_sents=None
        else:
            concat_embeddings, bert_sents=self.embedding_model(src, segs, clss, mask_src, mask_cls)

        sent_results = [classifier(concat_embeddings,mask_cls) for classifier in self.classifier_sup]
        sent_scores=[result[0] for result in sent_results]
        sent_probs=[result[1] for result in sent_results]
        
        if self.args.doc_classifier == 'FC' or self.args.doc_classifier == 'Probabilistic':
            if self.args.nbr_class_neurons==1:
                first_prob = (torch.sigmoid(sent_scores[0][...,0])*mask_cls)
                second_prob = (torch.sigmoid(sent_scores[1][...,0])*mask_cls)
            else:
                first_prob = (torch.nn.functional.softmax(sent_scores[0][:,:,:2],dim=-1)*mask_cls.unsqueeze(-1))[:,:,1]
                second_prob = (torch.nn.functional.softmax(sent_scores[1][:,:,:2],dim=-1)*mask_cls.unsqueeze(-1))[:,:,1]
            max_values_first,max_first = torch.sort(first_prob)
            max_values_second, max_second = torch.sort(second_prob)
            max_values= torch.cat((max_values_first[:,-1:],max_values_second[:,-1:]),axis=-1)
            max_all_pos = torch.cat((max_first[:,-1:],max_second[:,-1:]),axis=-1)
            d1 = concat_embeddings.size(0)
            indices=(torch.arange(d1).unsqueeze(1).repeat((1, max_all_pos.size(1))).flatten(),max_all_pos.flatten())
            best_sentences = concat_embeddings[indices].view(d1,max_all_pos.size(1),-1)
            contradiction_scores, contradiction_probs, output_embeddings = self.doc_classifier(best_sentences, mask_cls)
            new_co_no_co_labels=None
        else:
            contradiction_scores, contradiction_probs, output_embeddings = self.doc_classifier(concat_embeddings, mask_cls)
            best_sentences = None
            max_values = None
            new_co_no_co_labels=None

        # For contradictions analysis    
        if inference:
            with torch.no_grad():
                if self.args.nbr_class_neurons==1:
                    first_prob = (torch.sigmoid(sent_scores[0][...,0])*mask_cls)
                    second_prob = (torch.sigmoid(sent_scores[1][...,0])*mask_cls)
                else:
                    first_prob = (torch.nn.functional.softmax(sent_scores[0][...,:2],dim=-1)*mask_cls.unsqueeze(-1))[...,1]
                    second_prob = (torch.nn.functional.softmax(sent_scores[1][...,:2],dim=-1)*mask_cls.unsqueeze(-1))[...,1]
                max_values_first,max_first = torch.sort(first_prob, descending=True)
                max_values_second, max_second = torch.sort(second_prob, descending=True)
                max_first=max_first[:,:self.args.nbr_contradictions].unsqueeze(1)
                max_second=max_second[:,:self.args.nbr_contradictions].unsqueeze(1)
                best_contradictions=torch.cat((max_first,max_second),dim=1).transpose(1,2)
                if self.args.doc_classifier == 'Probabilistic':
                    best_probs=max_values_first[:,0]*max_values_second[:,0]
                else:
                    if self.args.nbr_class_neurons==1:
                        best_probs = torch.sigmoid(contradiction_scores[...,0])
                    else:
                        best_probs=torch.nn.functional.softmax(contradiction_scores[...,:2],dim=-1)[...,1]
            return sent_scores, sent_probs, contradiction_scores, contradiction_probs, new_co_no_co_labels, best_contradictions, best_probs, concat_embeddings, bert_sents, output_embeddings, best_sentences, max_values

        return sent_scores, sent_probs, contradiction_scores, contradiction_probs, new_co_no_co_labels, concat_embeddings, bert_sents, output_embeddings, best_sentences, max_values
       
class Generator(nn.Module):
    """docstring for ClassName"""
    def __init__(self,args, size_embedding, device, checkpoint):
        super(Generator,self).__init__()
        self.args = args
        self.device = device
        self.size_embedding= size_embedding
        self.num_sentences = 10
        if self.args.generator=='LSTM_doc' and self.args.use_output:
            self.input_size = self.args.random_input_size*2
        else:
            self.input_size = self.args.random_input_size

        if self.args.generator == 'FC_sent':
            self.gen_model = nn.Sequential(
            nn.Linear(self.args.random_input_size, self.args.size_hidden_layers),
            nn.ReLU(),        
            nn.Linear(self.args.size_hidden_layers, self.args.size_hidden_layers),
            nn.ReLU(),
            nn.Linear(self.args.size_hidden_layers, self.args.size_hidden_layers),
            nn.ReLU(),    
            nn.Linear(self.args.size_hidden_layers, self.size_embedding))
        elif self.args.generator == 'LSTM_sent':
            self.gen_model= nn.LSTM(self.args.random_input_size, 
                                    self.args.size_hidden_layers, 
                                    self.args.num_layers, 
                                    bidirectional=True)
            self.output_layer= nn.Sequential(nn.Linear(self.args.size_hidden_layers*2,
                                                       self.args.size_embedding))
        elif self.args.generator == 'Transformer_sent':
            self.gen_model= ExtTransformerEncoder(768, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, self.args.num_layers)
            self.output_layer= nn.Sequential(nn.Linear(self.size_embedding,
                                         self.size_embedding))
        elif self.args.generator == 'LSTM_doc':
            self.prep_layer= nn.Sequential(
                            nn.Linear(self.input_size, self.num_sentences*self.args.size_embedding),
                            nn.Sigmoid())
            self.gen_model= nn.LSTM(self.args.random_input_size, 
                                    self.args.size_hidden_layers, 
                                    self.args.num_layers, 
                                    bidirectional=True)
            self.output_layer= nn.Sequential(nn.Linear(self.args.size_hidden_layers*2,
                                                       self.args.size_embedding))
        elif self.args.generator == 'Transformer_doc':
            self.prep_layer= nn.Sequential(
                            nn.Linear(self.input_size, self.num_sentences*self.args.size_embedding),
                            nn.Sigmoid())
            self.gen_model= ExtTransformerEncoder(768, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, self.args.num_layers)
            self.output_layer= nn.Sequential(nn.Linear(self.size_embedding,
                                         self.size_embedding))

        if checkpoint is not None:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k[0:6]=='module':
                    name = k[7:] # remove module.
                else:
                    name=k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=True)

        self.to(self.device)

    def forward(self, random_input, mask=None):
        if self.args.generator == 'FC_sent':
            return self.gen_model(random_input)
        elif self.args.generator == 'LSTM_sent':
            input_sizes = list(random_input.size())
            self.gen_model.flatten_parameters()
            # h_0 = torch.empty((self.args.num_layers*2,input_sizes[0],self.args.size_hidden_layers)).normal_(mean=self.args.mean,std=self.args.std).to(random_input.get_device())
            # c_0 = torch.empty((self.args.num_layers*2,input_sizes[0],self.args.size_hidden_layers)).normal_(mean=self.args.mean,std=self.args.std).to(random_input.get_device())
            output_embeddings, (hn, cn) = self.gen_model(random_input.transpose(0,1))
            output_embeddings = output_embeddings.transpose(0,1)
            return self.output_layer(output_embeddings)
        elif self.args.generator == 'Transformer_sent':
            input_embeddings = self.gen_model(random_input, mask)
            return self.output_layer(input_embeddings)
        elif self.args.generator == 'LSTM_doc':
            input_embeddings=self.prep_layer(random_input).view(-1,self.num_sentences,self.args.size_embedding)
            input_sizes = list(input_embeddings.size())
            self.gen_model.flatten_parameters()
            # h_0 = torch.empty((self.args.num_layers*2,input_sizes[0],self.args.size_hidden_layers)).normal_(mean=self.args.mean,std=self.args.std).to(random_input.get_device())
            # c_0 = torch.empty((self.args.num_layers*2,input_sizes[0],self.args.size_hidden_layers)).normal_(mean=self.args.mean,std=self.args.std).to(random_input.get_device())
            output_embeddings, (hn, cn) = self.gen_model(random_input.transpose(0,1))
            output_embeddings = output_embeddings.transpose(0,1)
            return self.output_layer(output_embeddings)
        elif self.args.generator == 'Transformer_doc':
            input_embeddings=self.prep_layer(random_input).view(-1,self.num_sentences,self.args.size_embedding)
            mask= (input_embeddings.sum(dim=-1)!=0).to(input_embeddings.get_device())
            output_embeddings = self.gen_model(input_embeddings, mask)
            return self.output_layer(output_embeddings)