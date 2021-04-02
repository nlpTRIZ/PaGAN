# -*- coding: utf-8 -*-

import glob
import json
import os
import time
import subprocess
from os.path import join as pjoin
import re
import sys
import shutil
import random
import logging

#Data augmentation
#####
# from google.cloud import translate_v2 as translate
# # Instantiates a client
# translate_client = translate.Client()
import nlpaug.augmenter.word as naw
#####

from multiprocessing import Pool
from preprocessing.preprocess_init import *
from preprocessing.preprocess_nn import *
from preprocessing.utils import clean_text
from others.data_analysis import *



class Preprocessor:

    def __init__(self,args):

        # self.data_path : with which dataset to proceed (none = all)
        # self.temp_path : temporary directory to store output of Stanford nlp tokenizer
        # self.args.save_path_prepro : path to save preprocessing results
        # self.args.n_cpus : number cpus
        # self.parts_of_interest : which parts of the patents must be preprocessed

        self.temp_path=args.dataset_dir + '/TEMP'
        self.parts_of_interest = [str(item) for item in args.parts_of_interest.split(',')]
        self.args = args

        if args.mode =='test':
            self.data_path = args.input_directory
            dir_list = [args.dataset_dir,self.data_path,args.save_path_dir,args.save_path_nn,self.args.save_path_prepro,self.temp_path]
        else:
            self.data_path = [args.dataset_dir+'/'+dataset for dataset in args.dataset]
            dir_list=[args.dataset_dir]+self.data_path+[args.save_path_dir,args.save_path_nn,self.args.save_path_prepro,self.temp_path]

        for path_dir in [self.args.save_path_nn, self.args.save_path_prepro]:
            if os.path.isdir(path_dir):
                shutil.rmtree(path_dir)

        # Creation of all needed directories 
        for path in dir_list:
            try:
                os.mkdir(path)
            except:
                continue




    def tokenize(self):

        # This method rewrite the TRIZ summary in accordance with the will of the user (only one part of the contradiction in the summary or both parts)
        # It also rewrites every part involved to add spaces between sentences and avoid problems during tokenization
        # Finally it tokenizes text+summary using Stanford Core NLP. The results are saved 
       
        if self.args.mode == 'test':
            print("\n\n\nTokenization in progress...")
            if self.args.input_files is not None:
                files = self.args.input_files 
            else:
                addon = '/*'*(self.args.depth_directory+1)
                files=sorted(glob.glob(self.data_path+addon+self.args.input))

            count = 0
            pool = Pool(self.args.n_cpus)
            for _ in pool.imap_unordered(clean_text, files):
                 # Print % processed files
                #########################################################
                sys.stdout.write('\r')
                # the exact output you're looking for:
                j=(count+1)/len(files)
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                sys.stdout.flush()
                count+=1
                #########################################################

            pool.close()
            pool.join()

            extracted_patents_dir = os.path.abspath(self.data_path)
            tokenized_patents_dir = os.path.abspath(self.temp_path+'/test/'+self.args.input)
         
            print("Preparing to tokenize %s to %s..." % (extracted_patents_dir, tokenized_patents_dir))

            # make IO list file
            print("Making list of files to tokenize...")
            with open("mapping_for_corenlp.txt", "w") as f:
                for s in files:
                    f.write("%s\n" % (s))
            command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                       '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                       'json', '-outputDirectory', tokenized_patents_dir]
            print("Tokenizing %i files in %s and saving in %s..." % (len(files), extracted_patents_dir, tokenized_patents_dir))
            subprocess.call(command)
            
            os.remove("mapping_for_corenlp.txt")

            print("Stanford CoreNLP Tokenizer has finished.")
            print("Successfully finished tokenizing %s to %s.\n" % (extracted_patents_dir, tokenized_patents_dir))

        else:
            pool = Pool(self.args.n_cpus)
            for path in self.data_path:
                corpus_type = path.split('/')[-1]

                patents_directories = sorted(glob.glob(path+'/*'))
                for patent_directory in patents_directories:
                    found_summary=False
                    patent_files = sorted(glob.glob(patent_directory+'/*'))
                    for patent_file in patent_files:
                        if patent_file.find('.SUM')>=0 and patent_file.find('SUMMARY')<0:
                            found_summary=True
                    if not found_summary:
                        with open(patent_directory+'/'+os.path.basename(patent_directory)+'.SUM','w') as f:
                            f.write('')


                # Data augmentation with double translation
                # Add spaces between sentences in all used texts including summaries to avoid problems during tokenization
                #########################################################################################################################
                
                to_augment=[]
                count_augment = 0
                
                for tipe in ["SUM"]+self.parts_of_interest:
                    files=sorted(glob.glob(path+'/*/*.'+tipe))
                    count=0

                    if tipe != "SUM":
                        print()
                        logging.info("Cleaning data...")
                        count = 0
                        
                        for _ in pool.imap_unordered(clean_text, files):
                             # Print % processed files
                            #########################################################
                            sys.stdout.write('\r')
                            # the exact output you're looking for:
                            j=(count+1)/len(files)
                            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                            sys.stdout.flush()
                            count+=1
                            #########################################################
                    
                    if self.args.data_augmentation != "None" and corpus_type=='train':
                        print("\n"+corpus_type +" set: Data augmentation in progress for "+tipe+" parts...")
                    else:
                        print("\n"+corpus_type +" set: Sentences verification for SUM parts...")

                    for num_file,file in enumerate(files):

                        if tipe== "SUM":
                            with open(file, "r", encoding='utf-8') as f:
                                # print(file)
                                data=''
                                for sentence in f:
                                    if sentence.find("STATE_OF_THE_ART")>=0:
                                        continue
                                    else:
                                        data+= sentence

                                param_sents = data.split("///")
                                if param_sents[0]!='empty':
                                    to_augment.append(num_file)
                                first_param_sents = param_sents[0].replace("\n","").split("//")
                                if len(param_sents)>1:
                                    second_param_sents = param_sents[1].replace("\n","").split("//")
                                else:
                                    second_param_sents=[]

                        # Print % processed files
                        #########################################################
                        sys.stdout.write('\r')
                        # the exact output you're looking for:
                        j=(count+1)/len(files)
                        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                        sys.stdout.flush()
                        count+=1
                        #########################################################

                        # Data augmentation with double translation
                        if self.args.data_augmentation != "None" and corpus_type=='train' and num_file in to_augment:

                            count_augment+=1
                            if self.args.data_augmentation=="transformation" and self.args.transformation_type=="bert_embeddings":
                                aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
                            elif self.args.data_augmentation=="transformation" and self.args.transformation_type=="word2vec_embeddings":
                                aug = naw.WordEmbsAug(model_type='word2vec', model_path='./word2vec/GoogleNews-vectors-negative300.bin')
                            elif self.args.data_augmentation=="transformation" and self.args.transformation_type=="synonyms":
                                aug = naw.SynonymAug()


                            path_augmented_text = file.split('/')
                            path_augmented_text[-2]+='b'
                            path_augmented_text[-1]='.'.join([path_augmented_text[-2],path_augmented_text[-1].split('.')[-1]])
                            path_new_directory = '/'.join(path_augmented_text[:-1])
                            path_augmented_text='/'.join(path_augmented_text)

                            if os.path.isfile(path_augmented_text) or file.find('b')>0:
                                continue

                            if self.args.data_augmentation=="translation":
                                # Not to exceed google translations quotas
                                time.sleep(1.25)

                            augmented_text =''
                            if tipe != "SUM":
                                for sentence in data.split('.'):
                                    if self.args.data_augmentation=="translation":
                                        augmented_text += translate_client.translate(translate_client.translate(sentence+'.',target_language=self.args.translation_language)['translatedText'].replace("&#39;","'").replace("."," ")+'.',target_language='en')['translatedText'].replace("&#39;","'").replace("."," ")+'.'
                                    elif self.args.data_augmentation=="transformation":
                                        augmented_text += aug.augment(sentence+'.').replace("."," ")+'.'
                                        # print("ok2")
                            elif first_param_sents[0]!='empty':
                                for sentence in first_param_sents:
                                    if self.args.data_augmentation=="translation":
                                        augmented_text += translate_client.translate(translate_client.translate(sentence+'.',target_language=self.args.translation_language)['translatedText'].replace("&#39;","'").replace("."," ")+'.',target_language='en')['translatedText'].replace("&#39;","'").replace("."," ")+'. //'
                                    elif self.args.data_augmentation=="transformation":
                                        augmented_text += aug.augment(sentence+'.').replace("."," ")+'. //'
                                        # print("ok3")
                                augmented_text +='/'
                                for sentence in second_param_sents:
                                    if self.args.data_augmentation=="translation":
                                        augmented_text += translate_client.translate(translate_client.translate(sentence+'.',target_language=self.args.translation_language)['translatedText'].replace("&#39;","'").replace("."," ")+'.',target_language='en')['translatedText'].replace("&#39;","'").replace("."," ")+'. //'
                                    elif self.args.data_augmentation=="transformation":
                                        augmented_text += aug.augment(sentence+'.').replace("."," ")+'. //'
                                        # print("ok4")
                                augmented_text = augmented_text[:-3]


                            augmented_text = augmented_text.replace(".",". ")
                            augmented_text = ' '.join(augmented_text.split())


                            # Write translation
                            try:
                                os.mkdir(path_new_directory)
                            except:
                                pass

                            with open(path_augmented_text,'w') as f:
                                f.write(augmented_text[:-1])


                #########################################################################################################################
                # Rewriting of summaries with chosen sentences/parameters (one side of the contradiction or both)
                #########################################################################################################################""
                data_analyzer= summary_preparation(path+'/')

                (path_state_of_the_art,summary) = data_analyzer.get_data('both')
                for num in range (0,len(summary[0])):
                    summary_patent_first = ''
                    summary_patent_second = ''
                    for x in range(0,len(summary[0][num])):
                        summary_patent_first += (summary[0][num][x]+' ')
                    with open(path_state_of_the_art[num][0:-16]+'SUMTRIZ', "w") as file:
                        file.write(summary_patent_first)

                    for x in range(0,len(summary[1][num])):
                        summary_patent_second += (summary[1][num][x]+' ') 
                    with open(path_state_of_the_art[num][0:-16]+'SUMTRIZ2', "w") as file:
                        file.write(summary_patent_second)
                   
                # except:
                #     print("No summaries provided for "+corpus_type+' files.')
                #     time.sleep(1)
                #########################################################################################################################""


                # Tokenization using Standford Core NLP
                #########################################################################################################################

                add_on = ["SUMTRIZ", "SUMTRIZ2"]


                for tipe in self.parts_of_interest+add_on:
                    print("\n\n\nTokenization in progress...")
                    files=sorted(glob.glob(path+'/*/*.'+tipe))
                    print(str(len(files))+" "+tipe+" found for "+corpus_type+" set.")

                    extracted_patents_dir = os.path.abspath(path)
                    tokenized_patents_dir = os.path.abspath(self.temp_path+'/'+corpus_type+'/'+tipe)
                 
                    print("Preparing to tokenize %s to %s..." % (extracted_patents_dir, tokenized_patents_dir))
                    stories = os.listdir(extracted_patents_dir)
                    # make IO list file
                    print("Making list of files to tokenize...")
                    with open("mapping_for_corenlp.txt", "w") as f:
                        for s in files:
                            f.write("%s\n" % (s))
                    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                               'json', '-outputDirectory', tokenized_patents_dir]
                    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), extracted_patents_dir, tokenized_patents_dir))
                    subprocess.call(command)
                    
                    os.remove("mapping_for_corenlp.txt")

                #########################################################################################################################

            print("Stanford CoreNLP Tokenizer has finished.")
            print("Successfully finished tokenizing %s to %s.\n" % (extracted_patents_dir, tokenized_patents_dir))

            pool.close()
            pool.join()




    
    def format_to_lines(self):

        # This method rewrites the output of Stanford Core NLP tokenizer into a single file with several source texts and their summaries

        print("\nJSON files simplification...")

        if self.args.mode == 'test':
            corpus_type = 'test'
            files=[]

            files=sorted(glob.glob(self.temp_path+'/'+corpus_type+'/'+self.args.input+'/*'+'json'))
            print("\n"+str(len(files))+" "+self.args.input+" found for "+corpus_type+" set.")
                
            # creation iterator for data
            a_lst = [(f) for f in files]
           
            pool = Pool(self.args.n_cpus)
            dataset = []
            p_ct = 0
            count=0

            for d in pool.imap_unordered(format_to_lines_, zip(a_lst)):
                dataset.append(d)
                if (len(dataset) > 999):
                    pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path_prepro+'/'+self.args.input, corpus_type, p_ct)
                    with open(pt_file, 'w') as save:
                        # save.write('\n'.join(dataset))
                        save.write(json.dumps(dataset))
                        p_ct += 1
                        dataset = []

                # Print % processed files
                #########################################################
                sys.stdout.write('\r')
                # the exact output you're looking for:
                j=(count+1)/len(files)
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                sys.stdout.flush()
                count+=1
                #########################################################


            pool.close()
            pool.join()
            if (len(dataset) > 0):
                pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path_prepro+'/'+self.args.input, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        else:
            for corpus_type in self.args.dataset:

                files=[]
                for tipe in self.parts_of_interest:
                    files=sorted(glob.glob(self.temp_path+'/'+corpus_type+'/'+tipe+'/*'+'json'))
                    summaries = sorted(glob.glob(self.temp_path+'/'+corpus_type+'/'+'SUMTRIZ'+'/*'+'json'))
                    
                    # # Shuffle files list
                    random.seed(1)
                    random.shuffle(files)

                    # # Shuffle summaries list with same seed
                    random.seed(1)
                    random.shuffle(summaries)

                    
                    summaries_2 = sorted(glob.glob(self.temp_path+'/'+corpus_type+'/'+'SUMTRIZ2'+'/*'+'json'))
                    random.seed(1)
                    random.shuffle(summaries_2)


                    print("\n"+str(len(files))+" "+tipe+" found for "+corpus_type+" set.")
                        
                    # creation iterator for data
                    a_lst = [(f) for f in files]
                    s_lst = [(summary) for summary in summaries]

                    
                    s_lst2 = [(summary) for summary in summaries_2]
                    gen_data = zip(a_lst,s_lst,s_lst2)
                   
                    pool = Pool(self.args.n_cpus)
                    dataset = []
                    p_ct = 0
                    count=0

                    for d in pool.imap_unordered(format_to_lines_, gen_data):
                        dataset.append(d)
                        if (len(dataset) > 999):
                            pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path_prepro+'/'+tipe, corpus_type, p_ct)
                            with open(pt_file, 'w') as save:
                                # save.write('\n'.join(dataset))
                                save.write(json.dumps(dataset))
                                p_ct += 1
                                dataset = []

                        # Print % processed files
                        #########################################################
                        sys.stdout.write('\r')
                        # the exact output you're looking for:
                        j=(count+1)/len(files)
                        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                        sys.stdout.flush()
                        count+=1
                        #########################################################


                    pool.close()
                    pool.join()
                    if (len(dataset) > 0):
                        pt_file = "{:s}.{:s}.{:d}.json".format(self.args.save_path_prepro+'/'+tipe, corpus_type, p_ct)
                        with open(pt_file, 'w') as save:
                            # save.write('\n'.join(dataset))
                            save.write(json.dumps(dataset))
                            p_ct += 1
                            dataset = []

        shutil.rmtree(self.temp_path)
        print('\n')

        


    def format_to_nn(self):

        # This method formats the preprocessed files by format_to_lines into files usable with a nn algorithm

        if self.args.mode == 'test':
            self.args.dataset = ['test']

        for corpus_type in self.args.dataset:

            a_lst = []
            for json_f in sorted(glob.glob(pjoin(self.args.save_path_prepro, '*' + corpus_type + '.*.json'))):
                real_name = json_f.split('/')[-1]
                a_lst.append((json_f, self.args, pjoin(self.args.save_path_nn, real_name.replace('json', 'pt'))))

            
            pool = Pool(self.args.n_cpus)

            for d in pool.imap(format_to_bert_, a_lst):
                pass

            pool.close()
            pool.join()

        shutil.rmtree(self.args.save_path_prepro)

            # elif args.model == 'neusum':
            #     src_path = pjoin(args.save_path_nn+args.model,'text_'+corpus_type+'.src.txt')
            #     tgt_path = pjoin(args.save_path_nn+args.model,'text_'+corpus_type+'.tgt.txt')

            #     writer_src = open(src_path, 'w', encoding='utf-8')
            #     writer_tgt = open(tgt_path, 'w', encoding='utf-8')

            #     for json_f in sorted(glob.glob(pjoin(args.save_path_prepro, '*' + corpus_type + '.*.json'))):
            #         src,tgt = format_to_neusum(json_f)
            #         writer_src.write(src)
            #         writer_tgt.write(tgt)

            #     writer_src.close()
            #     writer_tgt.close()

            #      # Verification if some labels are empty (2nd part of contradiction)
            #     ############################################################################################################################
            #     src=''
            #     tgt=''

            #     with open(src_path, 'r', encoding='utf-8') as src_reader, \
            #         open(tgt_path, 'r', encoding='utf-8') as tgt_reader:
            #             for src_line, tgt_line in zip(src_reader, tgt_reader):
            #                 if tgt_line !='\n':
            #                     tgt+=tgt_line
            #                     src+=src_line
            #     src_writer = open(src_path, 'w', encoding='utf-8')
            #     tgt_writer = open(tgt_path, 'w', encoding='utf-8')
            #     src_writer.write(src)
            #     tgt_writer.write(tgt)
            #     src_writer.close()
            #     tgt_writer.close()
            #     ############################################################################################################################

            #     # Oracle computation
            #     oracle_path = pjoin(args.save_path_nn+args.model,corpus_type+'.rouge_bigram_F1.oracle')
            #     oracle_rouge_path = pjoin(args.save_path_nn+args.model,corpus_type+'rouge_bigram_F1.oracle.regGain')

            #     print('\n'+corpus_type+' set: Oracle predictions...')
            #     find_oracle(src_path,tgt_path,oracle_path,50,100000)

            #     print(corpus_type+' set: Regression gains computation...\n')
            #     get_regression_gain(src_path, tgt_path, oracle_path, oracle_rouge_path)



            


