import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward

class MLP_classifier(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.args = args
        self.output_size = self.args.nbr_class_neurons
        
        if self.args.gan_mode or (not self.args.gan_mode and not self.args.cnn_daily_mail_model):
            self.output_size +=1
        self.prob_layer=nn.Softmax(dim=-1)

        self.model = nn.Sequential(nn.Linear(num_features, num_features//4),
            nn.ReLU(),
            nn.Linear(num_features//4, self.output_size))

    def forward(self, input_tensor, mask_cls):
        logits= self.model(input_tensor)*mask_cls.unsqueeze(-1)
        probas = self.prob_layer(logits)*mask_cls.unsqueeze(-1)
        return logits, probas

class doc_classifier(nn.Module):
    def __init__(self, num_features, n_hid_lstm=768, num_layers=1, device=None, use_packed_data=False, doc_classifier=None, args=None):
        super().__init__()

        self.device = device
        self.doc_classifier=doc_classifier
        self.n_hid_lstm = n_hid_lstm
        self.num_layers = num_layers
        self.use_packed_data = use_packed_data
        self.intermediate_layers=True
        self.args=args
        self.output_size = self.args.nbr_class_neurons
        if self.args.gan_mode or (not self.args.gan_mode and not self.args.cnn_daily_mail_model):
            self.output_size +=1

        if self.doc_classifier == 'LSTM':
            self.net = nn.LSTM(num_features, n_hid_lstm, num_layers, bidirectional=False, dropout=0.2)
        elif self.doc_classifier == 'GRU':
            self.net = nn.GRU(num_features, n_hid_lstm, num_layers, bidirectional=False)
        elif self.doc_classifier == 'Transformer':
            self.net = ExtTransformerEncoder(768, args.ext_ff_size, args.ext_heads,
                                               0, 1)
        if self.doc_classifier == 'Transformer':
            self.model = nn.Sequential(nn.Linear(n_hid_lstm, self.output_size))
        elif self.doc_classifier == 'LSTM':
            self.model = nn.Sequential(nn.Linear(n_hid_lstm*2, self.output_size))
        else:
            if self.intermediate_layers:
                self.net_first = nn.Sequential(nn.Linear(768, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU())
                self.net_second = nn.Sequential(nn.Linear(768, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU())
                self.model = nn.Sequential(nn.Linear(128*2, self.output_size))
            else:
                self.model = nn.Sequential(nn.Linear(2*768,self.output_size))

        self.prob_layer=nn.Softmax(dim=-1)

    def forward(self, input_tensor, mask_cls):
        if self.doc_classifier not in ['FC', 'Probabilistic']:
            input_tensor = input_tensor*mask_cls.unsqueeze(-1)
        if self.doc_classifier == 'LSTM' or self.doc_classifier == 'GRU':
            # Get index of last non zeros elmts
            last_non_zero_index = (input_tensor.sum(dim=-1)!=0).sum(dim=-1)-1

        if self.use_packed_data:
            self.net.flatten_parameters()
            sorted_, indices = torch.sort(last_non_zero_index,descending=True)
            single_inputs = torch.nn.utils.rnn.pack_sequence([input_tensor[indice][:sort+1] for (sort,indice) in zip(sorted_, indices)])
            if self.doc_classifier == 'LSTM':
                last_embeddings, (hn, cn) = self.net(single_inputs)
            else:
                output_embeddings, hn = self.net(single_inputs)
                unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output_embeddings)
                _, new_indices = torch.sort(indices)
                last_embeddings = unpacked.transpose(0,1)[new_indices]
        else:
            if self.doc_classifier == 'LSTM':
                self.net.flatten_parameters()
                output_embeddings, (hn, cn) = self.net(input_tensor.transpose(0,1))
                output_embeddings=output_embeddings.transpose(0,1)
                last_embeddings = output_embeddings[torch.arange(len(input_tensor)),last_non_zero_index]
                last_embeddings = torch.cat((last_embeddings,cn[0]),dim=-1)
            elif self.doc_classifier == 'GRU':
                self.net.flatten_parameters()
                output_embeddings, hn = self.net(input_tensor.transpose(0,1))
                output_embeddings=output_embeddings.transpose(0,1)
                # Get last non zero embedding
                last_embeddings= output_embeddings[torch.arange(len(input_tensor)),last_non_zero_index]
                last_embeddings = torch.cat((last_embeddings,hn[0]),dim=-1)
            elif self.doc_classifier == 'Transformer':
                last_embeddings=self.net(input_tensor,mask_cls)[:,0,:]
            else:
                if self.intermediate_layers:
                    first_part_rep = self.net_first(input_tensor[:,0,:])
                    second_part_rep = self.net_second(input_tensor[:,1,:])
                    last_embeddings = torch.cat((first_part_rep,second_part_rep),axis=-1)
                else:
                    last_embeddings = input_tensor.view(len(input_tensor),-1)

        logits = self.model(last_embeddings)
        probas = self.prob_layer(logits)

        return logits, probas, last_embeddings




class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None]
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
        return x


def c_sampling(embeddings, mask_cls, labels, nbr_random):
    
    # create contradictions embeddings couples
    if labels is not None:
        # if embeddings.size(1)<3:
        #     nbr_random-=2
        # else:
        #     nbr_random=3

        # print("embeddings",embeddings.size())
        # print("labels",labels)
        # print("mask_cls",mask_cls)
        unlabeled_limits=mask_cls.sum(dim=-1)
        # print("unlabeled_limits",unlabeled_limits)
        first_ind = torch.arange(0,embeddings.size(0)).repeat(nbr_random)
        random_first_part = (torch.rand(size=(nbr_random,unlabeled_limits.size(0))).to(unlabeled_limits.get_device())*unlabeled_limits).long().flatten()
        random_second_part = (torch.rand(size=(nbr_random,unlabeled_limits.size(0))).to(unlabeled_limits.get_device())*unlabeled_limits).long().flatten()
        # print("first",random_first_part.size())
        # print("second",random_second_part.size())
        rdm_embeddings_couples = torch.cat((embeddings[first_ind,random_first_part].unsqueeze(1),embeddings[first_ind,random_second_part].unsqueeze(1)),dim=1)
        # print("rdm_embeddings_couples",rdm_embeddings_couples.size())

        labels = labels.transpose(0,1).squeeze(dim=-1)
        # print("1",labels.size())
        nb1 = labels[0].sum(dim=-1)
        nb2 = labels[1].sum(dim=-1)

        if nb1.sum()>0:
            # if only first part, add identical second part
            result1=torch.zeros_like(nb1)
            result1[torch.where(nb1>0)]=1
            result2=torch.zeros_like(nb1)
            result2[torch.where(nb2>0)]=1
            rows_to_add = torch.where(result1-result2==1)[0]
            labels[1][rows_to_add]=labels[0][rows_to_add]
            nb1 = labels[0].sum(dim=-1)
            nb2 = labels[1].sum(dim=-1)

            # print("2",labels.size())

            pos_dataset_elmts_first = torch.where(labels[0]==1)
            pos_dataset_elmts_second = torch.where(labels[1]==1)
            first_dim_first, second_dim_first = pos_dataset_elmts_first
            first_dim_second, second_dim_second = pos_dataset_elmts_second
            # print('pos_dataset_elmts_first',pos_dataset_elmts_first)
            # print('pos_dataset_elmts_second',pos_dataset_elmts_second)

            # create all possible combinations
            first_dim_first=torch.repeat_interleave(first_dim_first, nb2.repeat_interleave(nb1), dim=0)
            second_dim_first=torch.repeat_interleave(second_dim_first, nb2.repeat_interleave(nb1), dim=0)
            first_dim_second=torch.repeat_interleave(first_dim_second, nb1.repeat_interleave(nb2), dim=0)
            # add variations for the column instead of repeating
            nb1=nb1[torch.where(nb1>0)]
            nb2=nb2[torch.where(nb2>0)]
            prec_sep=0
            for i,sep in enumerate(nb2):
                if i==0:
                    list_tensor=[second_dim_second[0:sep].repeat(nb1[i])]
                else:
                    list_tensor.append(second_dim_second[prec_sep:prec_sep+sep].repeat(nb1[i]))
                prec_sep +=sep
            
            second_dim_second= torch.cat(list_tensor)
            pos_dataset_elmts_first=(first_dim_first,second_dim_first)
            pos_dataset_elmts_second=(first_dim_second,second_dim_second)
            # print('pos_dataset_elmts_first',pos_dataset_elmts_first)
            # print('pos_dataset_elmts_second',pos_dataset_elmts_second)

            embeddings_couples = torch.cat((embeddings[pos_dataset_elmts_first].unsqueeze(1),embeddings[pos_dataset_elmts_second].unsqueeze(1)),dim=1)
            # print("embeddings_couples",embeddings_couples.size())
            select_pos=torch.randperm(len(embeddings_couples))[:4]
            selected_couples=embeddings_couples[select_pos]

            length_cpl = len(selected_couples)

            # verification random couples are not correct
            for embedding in rdm_embeddings_couples:
                if embedding not in embeddings_couples:
                    selected_couples=torch.cat((selected_couples,embedding.unsqueeze(0)))
            all_embeddings=selected_couples

            #update_labels
            new_labels=torch.zeros(len(all_embeddings)).to(all_embeddings.get_device())
            new_labels[:length_cpl]=1

            return all_embeddings, new_labels
        else:
            rdm_embeddings_couples = torch.cat((embeddings[first_ind,random_first_part].unsqueeze(1),embeddings[first_ind,random_second_part].unsqueeze(1)),dim=1)
            all_embeddings = rdm_embeddings_couples
            new_labels = torch.zeros(len(all_embeddings)).to(all_embeddings.get_device())
            # print("all_embeddings_dataset",all_embeddings.size())
            # print("dataset",new_labels.size())
            return all_embeddings, new_labels

    unlabeled_limits=mask_cls.sum(dim=-1)
    # print("unlabeled_limits",unlabeled_limits)
    first_ind = torch.arange(0,embeddings.size(0)).repeat(nbr_random)
    random_first_part = (torch.rand(size=(nbr_random,unlabeled_limits.size(0))).to(unlabeled_limits.get_device())*unlabeled_limits).long().flatten()
    random_second_part = (torch.rand(size=(nbr_random,unlabeled_limits.size(0))).to(unlabeled_limits.get_device())*unlabeled_limits).long().flatten()
    # print("first",random_first_part.size())
    # print("second",random_second_part.size())
    rdm_embeddings_couples = torch.cat((embeddings[first_ind,random_first_part].unsqueeze(1),embeddings[first_ind,random_second_part].unsqueeze(1)),dim=1)
    # print("rdm_embeddings_couples",rdm_embeddings_couples.size())
    all_embeddings = rdm_embeddings_couples
    new_labels = torch.ones(len(all_embeddings)).to(all_embeddings.get_device())*3

    return all_embeddings, new_labels


def c_all(embeddings, first_ind, second_ind, labels, test_all=False):
    list_combinations=[]
    embeddings_couples=[]
    if test_all:
        all_ind = torch.cat((first_ind,second_ind),dim=1)
        # first_ind = (torch.repeat_interleave(torch.arange(len(embeddings)),torch.ones(len(embeddings)).long()*all_ind.size(1)),all_ind.flatten())
        # print(first_ind)
        for dim in range(len(all_ind)):
            list_combinations.append(torch.unique(torch.combinations(all_ind[dim], with_replacement=True),dim=0))
        # print(list_combinations)
    else:
        for dim in range(len(first_ind)):
            # print(first_ind[dim])
            # print(second_ind[dim])
            # print([torch.cat((a.unsqueeze(0),b.unsqueeze(0))) for a in first_ind[dim] for b in second_ind[dim]])
            list_inter=[]
            embeddings_inter=[]
            for a in first_ind[dim]:
                for b in second_ind[dim]:
                    list_inter.append(torch.cat((a.unsqueeze(0),b.unsqueeze(0))).unsqueeze(0))
                    embeddings_inter.append(torch.cat((embeddings[dim,a].unsqueeze(0),embeddings[dim,b].unsqueeze(0))).unsqueeze(0))
            # print(embeddings_inter)
            embeddings_couples.append(torch.cat(embeddings_inter))
            list_combinations.append(torch.cat(list_inter))

        embeddings_couples=torch.cat(embeddings_couples)
        combination_indices=torch.cat(list_combinations).long()

    return embeddings_couples, combination_indices


        