import bisect
import gc
import glob
import random

import torch
from others.logging import logger

class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, mode='train', classification='multi_class'):
        """Create a Batch from a list of examples.
            pre_src: list of sentences embeddings (indices of words in dictionnary)
            src: normalized pre_src with all identical length across batch (padding with 0)
            pre_tgt: list of sentences embeddings (indices of words in dictionnary)
                     for first part of contradiction
            tgt: normalized pre_tgt with all identical length across batch (padding with 0)
            pre_tgt2: list of sentences embeddings (indices of words in dictionnary)
                      for second part of contradiction
            tgt2: normalized pre_tgt2 with all identical length (padding with 0)

            pre_segs: segment embeddings (0 or 1) to identify sentences
            segs: normalized pre_segs with all identical length across batch (padding with 0)

            pre_clss: indices of output representations (one output for each sentence)
            clss: normalized pre_clss with all identical length across batch (padding with 0)

            pre_src_sent_labels: list of sentences positions of labeled sentences for 
                                 first part of contradiction
            label_first: normalized pre_src_sent_labels with all identical length across 
                         batch (padding with 0)

            pre_src_sent_labels2: list of sentences positions of labeled sentences for 
                                  second part of contradiction
            label_second: normalized pre_src_sent_labels2 with all identical length across 
                          batch (padding with 0)

            is_contradiction: list of int (0 if no contradiction in doc, 1 if contradiction,
                              2 for generated docs, 3 for unlabeled docs)
        """
        if data is not None:
            self.batch_size = len(data)

            if mode == 'train':
                pre_src = [x[0] for x in data]
                # pre_tgt = [x[1] for x in data]
                src = torch.tensor(self._pad(pre_src, 0))
                # tgt = torch.tensor(self._pad(pre_tgt, 0))
                # pre_tgt2 = [x[2] for x in data]
                pre_segs = [x[3] for x in data]
                pre_clss = [x[4] for x in data]
                pre_src_sent_labels = [x[5] for x in data]
                pre_src_sent_labels2 = [x[6] for x in data]
                # tgt2 = torch.tensor(self._pad(pre_tgt2, 0))
                segs = torch.tensor(self._pad(pre_segs, 0))
                mask_src = ~(src == 0)
                # mask_tgt = ~(tgt == 0)
                clss = torch.tensor(self._pad(pre_clss, -1))
                label_first = torch.tensor(self._pad(pre_src_sent_labels, 0))
                label_second = torch.tensor(self._pad(pre_src_sent_labels2, 0))
                is_contradiction = torch.tensor([x[-2] for x in data]).int()
                liste_label=[label_first,label_second]

                src_sent_labels = []
                for i in range(2):
                    if classification=='multi_class':
                        src_sent_labels.append(torch.stack((label_first,label_second),dim=-1).unsqueeze(0))
                    elif classification=='first':
                        src_sent_labels.append(label_first.unsqueeze(-1).unsqueeze(0))
                    elif classification=='second':
                        src_sent_labels.append(label_second.unsqueeze(-1).unsqueeze(0))
                    elif classification=='separate':
                        src_sent_labels.append(liste_label[i].unsqueeze(-1).unsqueeze(0))
                    else:
                        label_second -= torch.logical_and((label_first==1),(label_second==1)).long()
                        src_sent_labels.append((label_first + 2*label_second).unsqueeze(-1).unsqueeze(0))
               
                src_sent_labels = torch.cat(src_sent_labels,dim=0)
                length_first_part = torch.sum((label_first>0).float(),dim=-1)
                length_second_part = torch.sum((label_second>0).float(),dim=-1)

                mask_cls = ~(clss == -1)
                clss[clss == -1] = 0

                setattr(self, 'is_contradiction', is_contradiction.to(device))
                setattr(self, 'clss', clss.to(device))
                setattr(self, 'mask_cls', mask_cls.to(device))
                setattr(self, 'src_sent_labels', src_sent_labels.to(device))
                setattr(self, 'src', src.to(device))
                # setattr(self, 'tgt', tgt.to(device))
                # setattr(self, 'tgt2', tgt2.to(device))
                setattr(self, 'segs', segs.to(device))
                setattr(self, 'mask_src', mask_src.to(device))
                # setattr(self, 'mask_tgt', mask_tgt.to(device))
                # setattr(self, 'length_first', length_first_part.to(device))
                # setattr(self, 'length_second', length_second_part.to(device))
                src_str = [x[-5] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-4] for x in data]
                setattr(self, 'tgt_str', tgt_str)
                tgt_str2 = [x[-3] for x in data]
                setattr(self, 'tgt_str2', tgt_str2)
                ref_patent = [x[-1] for x in data]
                setattr(self, 'ref_patent', ref_patent)


            else:
                pre_src = [x[0] for x in data]
                src = torch.tensor(self._pad(pre_src, 0))
                pre_segs = [x[1] for x in data]
                pre_clss = [x[2] for x in data]
                segs = torch.tensor(self._pad(pre_segs, 0))
                mask_src = ~(src == 0)
                clss = torch.tensor(self._pad(pre_clss, -1))
                mask_cls = ~(clss == -1)
                clss[clss == -1] = 0
                setattr(self, 'clss', clss.to(device))
                setattr(self, 'mask_cls', mask_cls.to(device))
                setattr(self, 'src', src.to(device))
                setattr(self, 'segs', segs.to(device))
                setattr(self, 'mask_src', mask_src.to(device))
                len_cut = [x[-4] for x in data]
                setattr(self, 'len_cut', len_cut)
                src_str = [x[-3] for x in data]
                setattr(self, 'src_str', src_str)
                indic_piece = [x[-2] for x in data]
                setattr(self, 'indic_piece', indic_piece)
                ref_patent = [x[-1] for x in data]
                setattr(self, 'ref_patent', ref_patent)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    if args.mode == 'test':
        args.bert_data_path = '/'.join(args.bert_data_path.split('/')[:-1])+'/'+args.input
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    src = new[0]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, shuffle=True):
        self.args = args
        self.batch_size, self.dataset = batch_size, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs


    def preprocess(self, ex):
        
        src = ex['src']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']

        if self.args.mode == 'train':
            tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
            tgt2 = ex['tgt2'][:self.args.max_tgt_len][:-1] + [2]
            src_sent_labels = ex['src_sent_labels']
            src_sent_labels2 = ex['src_sent_labels_2']
            tgt_txt = ex['tgt_txt']
            if tgt_txt == 'empty':
                is_contradition = 3
            elif tgt_txt == '':
                is_contradition = 0
            else:
                is_contradition = 1
            tgt_txt2 = ex['tgt_txt2']

            end_id = [src[-1]]
            src = src[:-1][:self.args.max_pos - 1] + end_id
            segs = segs[:self.args.max_pos]
            max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
            clss = clss[:max_sent_id]
            src_txt = src_txt[:max_sent_id]
            src_sent_labels = src_sent_labels[:max_sent_id]
            src_sent_labels2 = src_sent_labels2[:max_sent_id]

            try:
                ref_patent = ex['ref_patent']
            except:
                ref_patent = None

            return src, tgt, tgt2, segs, clss, src_sent_labels, src_sent_labels2, src_txt, tgt_txt, tgt_txt2, is_contradition, ref_patent

        else:

            # if len(src)>self.args.max_pos:
            #     print("Source text is too long !")

            cut_src = []
            cut_clss = []
            cut_segs = []

            end_cut = 0
            while (clss[end_cut]<(len(src)-self.args.max_pos)):
                new_end_cut = end_cut + bisect.bisect_left(clss[end_cut::], self.args.max_pos + clss[end_cut]) -1
                # if problem in file
                if new_end_cut==end_cut:
                    return
                cut_src.append(src[clss[end_cut]:clss[new_end_cut]])
                cut_clss.append(clss[end_cut:new_end_cut])
                cut_segs.append(segs[clss[end_cut]:clss[new_end_cut]])
                end_cut = new_end_cut
                
            if end_cut < clss[-1]:
                cut_src.append(src[clss[end_cut]::])
                cut_clss.append(clss[end_cut::])
                cut_segs.append(segs[clss[end_cut]::])

            for i in range(len(cut_segs)):
                cut_clss[i]=[clss-cut_clss[i][0] for clss in cut_clss[i]]
                if cut_segs[i][0]==1:
                    cut_segs[i]=[1-seg for seg in cut_segs[i]]

            len_cut = [len(cut_clss_i) for cut_clss_i in cut_clss]

            ref_patent = ex['ref_patent']
           
            return cut_src, cut_segs, cut_clss, len_cut, src_txt, ref_patent


    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex)

            if(ex is None):
                continue

            if self.args.mode == 'test':
                for i in range(len(ex[0])):
                    if i == len(ex[0])-1:
                        indic = -1
                    else:
                        indic = i

                    exi= ex[0][i], ex[1][i], ex[2][i], ex[3], ex[4],indic, ex[5]
                    minibatch.append(exi)
                    # size_so_far += self.batch_size_fn(exi, len(minibatch))
                    size_so_far += 1

                # the batch can be longer than asked if a patent is cut into pieces
                if size_so_far >= batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0

            else:
                minibatch.append(ex)
                # size_so_far += self.batch_size_fn(ex, len(minibatch))
                size_so_far += 1
                if size_so_far == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                # elif size_so_far > batch_size:
                #     yield minibatch[:-1]
                #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)

        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            # size_so_far += self.batch_size_fn(ex, len(minibatch))
            size_so_far += 1

            if self.args.mode == 'test' and ex[-2]==-1:
                if size_so_far >= batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
            elif self.args.mode != 'test':
                if size_so_far == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                # elif size_so_far > batch_size:
                #     yield minibatch[:-1]
                #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            # print("buffer",buffer)
            if self.args.mode == 'test':
                p_batch = self.batch(buffer, self.batch_size)
            else:
                #p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = self.batch(buffer, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.args.mode, self.args.classification)

                yield batch
            return
