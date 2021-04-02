# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import time
import statistics 



class data_analysis:


    def __init__(self,dataset_path):

        self.dataset_path=dataset_path
        self.summaries = []
        self.states_of_the_art = []
        self.dates = []
        self.first_param_sents_list = []
        self.second_param_sents_list = []


    def listing(self):

        self.summaries=sorted(glob.glob(self.dataset_path+'/*/*.SUM'))
        print("\nThe "+ self.summaries[0].split('/')[3]+" dataset contains "+str(len(self.summaries))+" patents.\n")

        self.states_of_the_art=sorted(glob.glob(self.dataset_path+'/*/*.STATE_OF_THE_ART'))
        if len(self.states_of_the_art)!=len(self.summaries):
            print("There might be a problem. There are %i summaries and %i states of the art.\n"%(len(self.summaries),len(self.states_of_the_art)))

        for patent in self.summaries:
            self.dates.append(int(patent.split('/')[-1][11:15]))


        # res = plt.hist(self.dates, color = 'red', bins = [x - 0.5 for x in range(min(self.dates),max(self.dates)+2)], range=(min(self.dates),max(self.dates)+1), edgecolor = 'yellow')
        # plt.xlabel('Patents dates')
        # plt.ylabel('Number')
        # plt.title('Temporal distribution of patents in the database')
        # plt.show()



    def param_analysis(self):
        
        for summary  in self.summaries:
            text = ""
            for sentence in open(summary,"r"):
                if sentence.find("STATE_OF_THE_ART")>=0:
                    continue
                else:
                    text+= sentence
                    

        
            param_sents = text.split("///")
            first_param_sents = param_sents[0].replace("\n","").split("//")

            if len(param_sents)>1:
                second_param_sents = param_sents[1].replace("\n","").split("//")
            else:
                second_param_sents=param_sents[0].replace("\n","").split("//")

            self.first_param_sents_list.append(first_param_sents)
            self.second_param_sents_list.append(second_param_sents)

        # # Statistics
        # length_first_params_sent = [len(self.first_param_sents_list[i]) for i in range(0,len(self.first_param_sents_list))]
        # length_second_params_sent = [len(self.second_param_sents_list[i]) for i in range(0,len(self.second_param_sents_list))]
        # unique_number=0
        # for i in range(0,len(length_second_params_sent)):
        #   if length_second_params_sent[i]==0:
        #       unique_number+=1

        # print("Mean number of sentences for originally improved parameters: %.3f " % ((sum(length_first_params_sent)-unique_number)/(len(length_second_params_sent)-unique_number)))
        # print("Mean number of sentences for worsened parameters in state of the arts methods: %.3f " % ((sum(length_second_params_sent)/(len(length_second_params_sent)-unique_number))))
        # print("Percentage of sentences containing both originally improved parameters and worsened parameters in state of the arts methods: %.3f" % (unique_number/len(self.summaries)*100) +"%\n")




    def get_data(self,info):

        path_state_of_the_art = self.states_of_the_art

        if info=='first':
            summary = [self.first_param_sents_list]
                
        elif info=='second':
            summary = [self.second_param_sents_list]
                
        else:
            summary = [self.first_param_sents_list,self.second_param_sents_list]

        return (path_state_of_the_art,summary)





def summary_preparation(dataset_path):
        
    data_analyzer = data_analysis(dataset_path)
    data_analyzer.listing()
    data_analyzer.param_analysis()

    return data_analyzer




