# -*- coding: utf-8 -*-

from xml.etree import ElementTree as ET
import re
import hashlib
import json


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def load_json(p, lower, content):
	# extract text from json
    source = []
    tgt = []
    tgt2 = []

    if content == 3:
        (a_lst,s_lst,s_lst2)=p
    elif content == 2:
        (a_lst,s_lst)=p
    else:
        a_lst=p[0]
    

    source_s=''
    for sent in json.load(open(a_lst))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        # print("toookkens",tokens)
        source.append(tokens)
        # source_s = [source_s + t for t in tokens][0]
        # print("source sr",source_s)
    ref_patent = a_lst.split('/')[-1].split('.')[0]

    source = [clean(' '.join(sent)).split() for sent in source]
       
    if content >1:
        for sent in json.load(open(s_lst))['sentences']:
            tokens = [t['word'] for t in sent['tokens']]
            if (lower):
                tokens = [t.lower() for t in tokens]

            tgt.append(tokens)
        tgt = [clean(' '.join(sent)).split() for sent in tgt]
        

    if content==3:
        for sent in json.load(open(s_lst2))['sentences']:
            tokens = [t['word'] for t in sent['tokens']]
            if (lower):
                tokens = [t.lower() for t in tokens]

            tgt2.append(tokens)
        tgt2 = [clean(' '.join(sent)).split() for sent in tgt2]
        return source, tgt, tgt2, ref_patent


    if content == 2:
        return source, tgt
    else:
        return source, ref_patent

        



def format_to_lines_(params):
    f = params

    if len(f) == 3:
        source, tgt, tgt2, ref_patent = load_json(f, True, len(f))
        return {'src': source, 'tgt': tgt, 'tgt2':tgt2, 'ref_patent':ref_patent}
    elif len(f) == 2:
        source, tgt = load_json(f, True, len(f))
        return {'src': source, 'tgt': tgt}
    else:
        source, ref_patent = load_json(f, True, len(f))
        return {'src': source, 'ref_patent':ref_patent}

    


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
