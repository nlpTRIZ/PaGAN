from tkinter.filedialog import askopenfilename
import tkinter.messagebox as messagebox
from tkinter import *
import tkinter.ttk as ttk
import tkinter.font as tkFont
import os
import shutil
from threading import Thread
import queue
import glob
import time
import numpy as np
import re
import os
import math
import subprocess
from sys import platform
from database.dbPatent import dbPatent
database = dbPatent('csip','csip')


FREEZING = False #True if .exe creation
if FREEZING:
    addon = 'lib/'
else:
    addon = ''


from test import apply_model
from preprocessing.preprocess_nn import greedy_selection


class App(Tk):
 
    def __init__(self):
        Tk.__init__(self)

        self.is_declared = False
        self.geometry("2000x1500")
        self.Frame = Frame(self,bg='light grey')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.Frame.master.title("Guarino Inc.'s Automatic Annotator")
        self.Frame.grid(sticky=E+W+S+N)
        
        self.Frame.grid_columnconfigure(0, weight=1)
        self.Frame.grid_columnconfigure(1, weight=1)
        self.Frame.grid_rowconfigure(0, weight=1)

        self.button_size = 12
        self.button_font = tkFont.Font(family="Helvetica", weight="bold", size=self.button_size)
        self.indic_size = 10
        self.indic_font = tkFont.Font(family="Helvetica", size=self.indic_size)
        self.scale_size = 10
        self.scale_font = tkFont.Font(family="Helvetica", size=self.scale_size)
        self.title_size = 18
        self.title_font = tkFont.Font(family="Times", size=self.title_size, weight="bold")
        self.selection_size = 13
        self.selection_font = tkFont.Font(family="Times", size=self.selection_size, weight="bold")
        self.text_size = 8
        self.text_font = tkFont.Font(family="Times", size=self.text_size, weight="bold")
        
        # Frame left
        #########################################################################################################################################
        #########################################################################################################################################

        self.frame_left = Frame(self.Frame, bg = "light grey", borderwidth=10)
        self.frame_left.grid(row=0, column=0,sticky=E+W+S+N)
        self.frame_left.grid_columnconfigure(0, weight=1)
        self.frame_left.grid_rowconfigure(1, weight=1)

        # Frame for the scales
        #########################################################################################################################################
        self.frame_scale = Frame(self.frame_left,bg='light grey')
        self.frame_scale.grid(row=0, column=0, sticky=E+W+S+N)

        self.max_scale = 20
        self.scale_first= Scale(self.frame_scale, orient='horizontal', from_=1, to=self.max_scale,
            resolution=1, relief = 'flat', sliderrelief='flat',  sliderlength=80, label="Highlighted sentences (1st part)", font= self.scale_font, bg='orange red')
        self.scale_first.pack(side=LEFT,fill=BOTH,expand=True)
        self.scale_second= Scale(self.frame_scale, orient='horizontal', from_=1, to=self.max_scale,
            resolution=1, relief = 'flat', sliderrelief='flat',  sliderlength=80, label="Highlighted sentences (2nd part)", font= self.scale_font, bg='RoyalBlue1')
        self.scale_second.pack(side=RIGHT,fill=BOTH,expand=True)
        #########################################################################################################################################

        # Frame for text area
        #########################################################################################################################################
        self.frame_text = Frame(self.frame_left,bg='light grey', borderwidth=7)
        self.frame_text.grid(row=1, column=0,sticky=E+W+S+N)

        self.area = Text(self.frame_text, wrap=WORD, font=self.text_font)
        self.area.pack(fill=BOTH,expand=True)
        self.area.tag_config('first_sentences', background='orange red')
        self.area.tag_config('second_sentences', background='RoyalBlue1')
        self.area.tag_config('both_sentences', background='MediumPurple1')
        #########################################################################################################################################

        #Frame for buttons under text area
        #########################################################################################################################################
        self.frame_buttons = Frame(self.frame_left,bg='light grey')
        self.frame_buttons.grid(row=2, sticky=E+W+S+N)
        self.frame_buttons.grid_columnconfigure(1, weight=1)

        # Information Label
        self.label = Entry(self.frame_buttons,font=self.indic_font, justify='center',relief='flat',bg='light grey', borderwidth=7)
        self.label.grid(row=0, column=0,columnspan=2,sticky=E+W+S+N)
        self.proba_label = Entry(self.frame_buttons,font=self.indic_font, justify='center',relief='flat',bg='light grey', borderwidth=7)
        self.proba_label.grid(row=0, column=2,sticky=E+W+S+N)

        # Buttons
        previous_btn = Button(self.frame_buttons, text="Previous Patent",font=self.button_font,command=self.previous_patent)
        previous_btn.grid(row=1, column=0,sticky=E+W+S+N)
        next_btn = Button(self.frame_buttons, text="Analyze", font=self.button_font,command=self.analyze)
        next_btn.grid(row=1, column=1,sticky=E+W+S+N)
        a_btn = Button(self.frame_buttons, text="Next Patent", font =self.button_font,command=self.next_patent)
        a_btn.grid(row=1, column=2,sticky=E+W+S+N)
        #########################################################################################################################################

        #Frame for selection data
        #########################################################################################################################################
        self.frame_data = Frame(self.frame_left,bg='light grey', borderwidth=10)
        self.frame_data.grid(row=3, sticky=E+W+S+N)
        self.frame_data.grid_columnconfigure(3, weight=1)
        self.label_number_patents = Label(self.frame_data, text = "Number of patents",font=self.text_font,bg='light grey')
        self.label_number_patents.grid(row=0, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)
        self.number_patents_entry = Entry(self.frame_data,font=self.indic_font, justify='center',relief='flat',bg='light grey')
        self.number_patents_entry.grid(row=1, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)
        self.number_patents_entry.delete(0, END)
        self.number_patents_entry.insert(END,'10')
        self.load_btn = Button(self.frame_data, text="Load patents", font =self.button_font,command=self.load_patents)
        self.load_btn.grid(row=1, column=3,columnspan=1,rowspan=1,sticky=E+W+S+N)

        self.label_year_patents = Label(self.frame_data, text = "Year patents",font=self.text_font,bg='light grey')
        self.label_year_patents.grid(row=0, column=0,columnspan=2,rowspan=1,sticky=E+W+S+N)

        years = [year for year in range(1984,2021)]
        years.reverse()
        # create the listbox (note that size is in characters)
        self.year_box = Listbox(self.frame_data, height=3, font=self.text_font, relief='flat')
        self.year_box.grid(row=1, column=0,columnspan=1,rowspan=1,sticky=E+W+S+N)

        # create a vertical scrollbar to the right of the listbox
        self.yscroll = Scrollbar(self.frame_data,command=self.year_box.yview, orient=VERTICAL)
        self.yscroll.grid(row=1, column=1, rowspan=1, sticky=N+S)
        self.year_box.configure(yscrollcommand=self.yscroll.set)

        # load the listbox with data
        for item in years:
            self.year_box.insert(END, item)

        def get_item(event):
            index = self.year_box.curselection()[0]
            self.year = self.year_box.get(index)

        self.year_box.bind('<ButtonRelease-1>', get_item)
        self.year = 2020


        


        # end frame left
        #########################################################################################################################################
        #########################################################################################################################################



        # Frame right
        #########################################################################################################################################
        #########################################################################################################################################
        self.frame_right = Frame(self.Frame, bg = "light grey", borderwidth=10)
        self.frame_right.grid(row=0, column=1, sticky=E+W+S+N)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(0, weight=0)
        self.frame_right.grid_rowconfigure(1, weight=2)
        self.frame_right.grid_rowconfigure(2, weight=2)
        self.frame_right.grid_rowconfigure(3, weight=0)

        # frame for label
        # row 0
        self.frame_label = Frame(self.frame_right)
        self.frame_label.grid(row=0, column=0,rowspan=1,sticky=E+W+S+N)
        Label(self.frame_label,font=self.title_font,text="Sentences selection for summary generation",bg='light grey').pack(side=TOP,fill=BOTH,expand=True)


        # frame for add summary first
        #########################################################################################################################################
        self.frame_add_first = Frame(self.frame_right,bg='light grey', borderwidth=10)
        self.frame_add_first.grid(row=1, column=0,sticky=E+W+S+N)
        self.frame_add_first.grid_columnconfigure(0, weight=1)
        self.frame_add_first.grid_rowconfigure(1, weight=4)
        self.frame_add_first.grid_rowconfigure(3, weight=1)
        self.frame_add_first.grid_rowconfigure(4, weight=1)

        # Add sentence first to summary
        Label(self.frame_add_first, text = "Selected sentence (1st part)",bg='orange red',font=self.selection_font).grid(row=0, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)
        self.first_box = Text(self.frame_add_first,wrap=WORD,bg='#f1c0b5',font=self.text_font) 
        self.first_box.grid(row=1, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)

        add_first = Button(self.frame_add_first, text="Add sentence to summary",font=self.selection_font,bg='orange red',command=self.add_sentence_first)
        add_first.grid(row=2, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)

        self.box_first = Listbox(self.frame_add_first, bg='#f1c0b5', font=self.text_font)  # You'll want to keep this reference as an attribute of the class too.
        self.box_first.grid(row=3, column=0,columnspan=2,rowspan=2,sticky=E+W+S+N)

        edit_first = Button(self.frame_add_first, text="Edit",font=self.button_font,bg='orange red',command=self.edit_sentence_first)
        edit_first.grid(row=3, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)

        remove_first = Button(self.frame_add_first, text="Remove",font=self.button_font,bg='orange red',command=self.delete_sentence_first)
        remove_first.grid(row=4, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)
        #########################################################################################################################################


        # frame for add summary second
        #########################################################################################################################################
        self.frame_add_second = Frame(self.frame_right,bg='light grey', borderwidth=10)
        self.frame_add_second.grid(row=2, column=0,sticky=E+W+S+N)
        self.frame_add_second.grid_columnconfigure(0, weight=1)
        self.frame_add_second.grid_rowconfigure(1, weight=4)
        self.frame_add_second.grid_rowconfigure(3, weight=1)
        self.frame_add_second.grid_rowconfigure(4, weight=1)

        # Add sentence first to summary
       
        Label(self.frame_add_second, text = "Selected sentence (2nd part)",bg='RoyalBlue1',font=self.selection_font).grid(row=0, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)
        self.second_box = Text(self.frame_add_second,wrap=WORD,bg='#B0C4DE', font=self.text_font) 
        self.second_box.grid(row=1, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)

        add_second = Button(self.frame_add_second, text="Add sentence to summary",font=self.selection_font,bg='RoyalBlue1',command=self.add_sentence_second)
        add_second.grid(row=2, column=0,columnspan=3,rowspan=1,sticky=E+W+S+N)

        self.box_second = Listbox(self.frame_add_second, bg='#B0C4DE', font=self.text_font)  # You'll want to keep this reference as an attribute of the class too.
        self.box_second.grid(row=3, column=0,columnspan=2,rowspan=2,sticky=E+W+S+N)

        edit_second = Button(self.frame_add_second, text="Edit",font=self.button_font,bg='RoyalBlue1',command=self.edit_sentence_second)
        edit_second.grid(row=3, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)

        remove_second = Button(self.frame_add_second, text="Remove",font=self.button_font,bg='RoyalBlue1',command=self.delete_sentence_second)
        remove_second.grid(row=4, column=2,columnspan=1,rowspan=1,sticky=E+W+S+N)
        #########################################################################################################################################

        # Frame button
        #########################################################################################################################################
        self.frame_button_second = Frame(self.frame_right,bg='light grey', borderwidth=10)
        self.frame_button_second.grid(row=3, column=0, sticky=E+W+S+N)
        # self.frame_button_second.grid_columnconfigure(0, weight=1)
        # self.frame_button_second.grid_columnconfigure(1, weight=1)
        self.open_btn = Button(self.frame_button_second, text="Open current patent directory", font=self.button_font, command=self.open_file_explorer)
        self.open_btn.pack(side=TOP,fill=BOTH,expand=True)
        #########################################################################################################################################

        # End frame right
        #########################################################################################################################################
        #########################################################################################################################################


        # Sentences probabilities
        self.outQueue1 = queue.Queue(maxsize=1)
        self.outQueue2 = queue.Queue(maxsize=1)

        self.first_sentences_probabilities = None
        self.second_sentences_probabilities = None
        self.summary = None
        self.predict_contradiction=True

        self.scale_value_first = None
        self.scale_value_second = None
        
        
        try:
            os.mkdir('../data_patents/input_data')
            os.mkdir('../data_patents/input_data/Extracted_txt_patents')
        except:
            pass

        ###################################################
        self.list_patents = database.get_patents(nb = 10, part = None, date = '2020')
        # print(self.list_patents)
        self.current_patent_index=None
        self.current_patent=None
        self.update_label()
        self.update_size_continuously()
        ###################################################

        self.is_extracted = True
        self._thread = None
        self.current_action = None
 
        self.is_declared = True
        if len(self.list_patents)>0:
            self.next_patent()

        

    def load_patents(self):
        try:
            number = int(self.number_patents_entry.get())
        except:
            number=10
        self.number_patents_entry.delete(0, END)
        self.number_patents_entry.insert(END,str(number))
        self.list_patents = database.get_patents(nb = number, date = self.year)
        self.current_patent_index=None
        self.current_patent=None
        self.next_patent(force=True)

    
    def _current_patent(self):
        ref = self.list_patents[self.current_patent_index]['REF PATENT']
        name_directory = '../data_patents/input_data/Extracted_txt_patents/'+ref+'/'
        
        try:
            os.mkdir(name_directory)
            for key in self.list_patents[self.current_patent_index].keys():
                if key!='_id' and key!='REF PATENT':
                    name_file = name_directory+ref+'.'+key
                    text = self.list_patents[self.current_patent_index][key]

                    with open(name_file, 'w', encoding='utf-8') as file:
                        if key == 'STATE_OF_THE_ART' and isinstance(text, list):
                            text = text[0]
                        elif isinstance(text, list):
                            text = '\n'.join(text)
                        file.write(text)
        except:
            print('Directory exists already')

        
        self.current_patent = name_directory+ref+'.STATE_OF_THE_ART'


    # Extraction process
    #################################################################################################
    def extract(self):
        # if no current action
        if self._thread is None:

            # Do you want to erase current extracted data ?
            if self.is_extracted and self.question_extraction() or not self.is_extracted:
                if os.path.isdir('./data/Extraction_results/'):
                    shutil.rmtree('./data/Extraction_results/')
                if os.path.isdir('./data/Preprocessing_results/'):
                    shutil.rmtree('./data/Preprocessing_results/')
                self._thread = Thread(target=self.launch_extraction)
                self._thread.start()
        else:
            # send message "an action is already performing"
            self.error_message()

 
    def launch_extraction(self):
        path_name = self.open_menu(initialdir='./data/Downloaded_data',filetype=[("Archive", ".zip"),("Archive", ".tar")])
        
        #if path_name exists
        if path_name != () and path_name != '':
            self.current_action = 'Extraction'
            self.indicate_thread()
            call_extractor_preprocessor(task = 'Extraction', path=path_name)
            # cmd = "./patent_extraction_preprocessing/main.py -target_file "+path_name
            # os.system('{} {}'.format('python3 -q -X faulthandler', cmd))
            self.is_extracted=True
        self.patent_list = [patent.replace('\\','/') for patent in sorted(glob.glob('./data/Extraction_results/Extracted_txt_patents/*/*.STATE_OF_THE_ART'))]

        

    def question_extraction(self):
        return messagebox.askquestion ("Guarino Inc.'s News Report",'Are you sure you want to delete your current extracted data ?',icon = 'warning')=='yes'



    # Change current patent
    #################################################################################################

    def choose_patent(self):
        if self._thread is None:
            if len(self.patent_list)>0:
                path_name = self.open_menu(initialdir='./data/Extraction_results/Extracted_txt_patents')

                #if path_name exists
                if path_name != () and path_name != '':
                    # Preprocessing state of the art
                    #########################################################################################################
                    if not os.path.exists('.'.join(path_name.split('.')[:-1])+'.SUM'):
                        with open('.'.join(path_name.split('.')[:-1])+'.SUM','w'):
                            pass
                        self.first_box.delete("1.0",END)
                        self.second_box.delete("1.0",END)
                        self.box_first.delete(0,END)
                        self.box_second.delete(0,END)
                    else:
                        self.read_existing_summary(path_name)
                        self.first_box.delete("1.0",END)
                        self.second_box.delete("1.0",END)  

                    # call_extractor_preprocessor(task ='Preprocessing', path=path_name)
                    #########################################################################################################

                    with open(path_name,'r', encoding='utf-8') as state_of_the_art:
                        text = ' '.join(re.sub(r"etc[.]\s?:[a-z]",'',state_of_the_art.read().replace('U.','').replace('S.','').replace('Pat.','').replace('Nos.','').replace('No.','').replace('etc.)','').replace('etc. )','').replace('etc. ,','').replace('e. g.','').replace('i. e.','')).split())
                    
                    # Update Text on screen
                    ############################################
                    self.area.delete(1.0, END)
                    self.area.insert(END,text)
                    ############################################

                    # Current patent and index update
                    #########################################################
                    self.current_patent='./'+'/'.join(path_name.split('/')[-5:])
                    self.current_patent_index = self.patent_list.index(self.current_patent)
                    self.update_label()
                    #########################################################

                    #No highlighted sentences / no sentences probabilities
                    #########################################################
                    with self.outQueue1.mutex:
                        self.outQueue1.queue.clear()
                    with self.outQueue2.mutex:
                        self.outQueue2.queue.clear()
                    self.first_sentences_probabilities = None
                    self.second_sentences_probabilities = None
                    self.summary = None
                    #########################################################
  
    
            else:
                self.need_extraction_message()
        else:
            # send message "an action is already performing"
            self.error_message()


    def next_patent(self,next_patent=True,force=False):
        if self._thread is None or force:
            if len(self.list_patents)>0:
                if self.current_patent is not None and not force:
                    
                    if next_patent: 
                        self.current_patent_index += 1
                    else:
                        self.current_patent_index -= 1
                    
                else:
                    self.current_patent_index = 0

                self.scale_first.set(1)
                self.scale_second.set(1)
                self.current_patent_index %=len(self.list_patents)
                self._current_patent()
                with open(self.current_patent,'r', encoding='utf-8') as state_of_the_art:
                    text = ' '.join(re.sub(r"etc[.]\s?:[a-z]",'',state_of_the_art.read().replace('U.','').replace('S.','').replace('Pat.','').replace('Nos.','').replace('No.','').replace('etc.)','').replace('etc. )','').replace('etc. ,','').replace('e. g.','').replace('i. e.','')).split())
                                
                # Update Text on screen
                ############################################
                self.area.delete(1.0, END)
                self.area.insert(END,text)
                ############################################

                                # Preprocessing state of the art
                #########################################################################################################
                if not os.path.exists('.'.join(self.current_patent.split('.')[:-1])+'.SUM'):
                    with open('.'.join(self.current_patent.split('.')[:-1])+'.SUM','w'):
                        pass
                    self.first_box.delete("1.0",END)
                    self.second_box.delete("1.0",END)
                    self.box_first.delete(0,END)
                    self.box_second.delete(0,END)
                else:
                    print("Reading existing summary")
                    self.read_existing_summary(self.current_patent)
                    self.first_box.delete("1.0",END)
                    self.second_box.delete("1.0",END)

                #call_extractor_preprocessor(task = 'Preprocessing',path='/'.join(patent_path.split('/')[:-1]))
                #########################################################################################################

                # Current patent and index update
                #########################################################
                self.update_label()
                self.update_proba_label(delete=True)
                #########################################################

                #No highlighted sentences / no sentences probabilities
                #########################################################
                with self.outQueue1.mutex:
                    self.outQueue1.queue.clear()
                with self.outQueue2.mutex:
                    self.outQueue2.queue.clear()
                self.first_sentences_probabilities = None
                self.second_sentences_probabilities = None
                self.summary = None
                #########################################################

                self.update_highlighted_sentences()

            else:
                self.need_extraction_message()
        else:
            self.error_message()


    def previous_patent(self):
        self.next_patent(next_patent=False)


    #################################################################################################
    def open_file_explorer(self):
        if self._thread is None:
            if len(self.list_patents)>0:
                if self.current_patent is not None:
                    # FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')
                    path = '/'.join(self.current_patent.split('/')[:-1])
                    # LINUX
                    if platform == "linux" or platform == "linux2":
                        os.system('xdg-open "%s"' % path)
                    # WINDOWS
                    elif platform =="win32":
                        path = '\\'.join(path.split('/'))
                        os.startfile(path)
                else:
                    self.choose_patent_message()
            else:
                self.need_extraction_message()
        else:
            self.error_message()


    def error_message(self):
        messagebox.showinfo("Guarino Inc.'s News Report",self.current_action+" in progress. Wait.")


    def need_extraction_message(self):
        messagebox.showinfo("Guarino Inc.'s News Report","You need to extract patents content first!")


    def already_did_this_message(self):
        messagebox.showinfo("Guarino Inc.'s News Report","You already did this.")


    def choose_patent_message(self):
        messagebox.showinfo("Guarino Inc.'s News Report","Choose a patent first.")

    
    def open_menu(self,initialdir,filetype = [("State of the Art", ".STATE_OF_THE_ART")]):
        path_name = askopenfilename(title="Select File", initialdir=initialdir, filetypes=filetype)
        return path_name


    def length_patent_list(self):
        if self.list_patents is not None:
            return len(self.list_patents)
        else:
            return None

    def update_label(self):
        if self.current_patent is not None and self.current_patent_index is not None:
            self.label.delete(0, END)
            self.label.insert(0, "Current patent: {}   Patent index: {} / {}".format(self.current_patent.split('/')[-1].split('.')[0],self.current_patent_index+1, self.length_patent_list()))
        else:
            self.label.delete(0, END)

    def update_proba_label(self, input=None, delete=False):
        if delete:
            self.proba_label.delete(0, END)
        else:
            self.proba_label.delete(0, END)
            self.proba_label.insert(0, "Pc = {:.1f}".format(input))



    def update_highlighted_sentences(self):
        self._thread2 = Thread(target=self.get_update_continuously)
        self._thread2.daemon = True
        self._thread2.start()


    def indicate_thread(self):
        self.label.delete(0, END)
        self.label.insert(0, "{} in progress...".format(self.current_action))


    def update_size_continuously(self):
        self._thread3 = Thread(target=self.update_size)
        self._thread3.daemon = True
        self._thread3.start()


    def update_size(self):
        height = 0
        width = 0
        while True:
            if math.sqrt((height-self.winfo_height())**2+(width-self.winfo_width())**2)>20:
                self.button_font.configure(size= min([int(self.button_size*self.winfo_width()/1500),self.button_size]))
                self.text_font.configure(size= min([int(self.text_size*self.winfo_width()/1500),self.text_size]))
                self.indic_font.configure(size= min([int(self.indic_size*self.winfo_width()/1800),self.indic_size]))
                self.scale_font.configure(size= min([int(self.scale_size*self.winfo_width()/2000),self.scale_size]))
                self.title_font.configure(size= min([int(self.title_size*self.winfo_width()/2000),self.title_size]))
                self.selection_font.configure(size= min([int(self.selection_size*self.winfo_width()/2000),self.selection_size]))
                height = self.winfo_height()
                width = self.winfo_width()


    def get_update_continuously(self):

        self.oracle_ids_first_last = None
        self.oracle_ids_second_last = None
        self._thread4=None
        self.oracle_ids_first=None
        self.oracle_ids_second=None
        self.valid=False
        self.indexes_list_first=[]
        self.indexes_list_second=[]
        
        while True:
            
            #try:
            #    widget = str(self.print_widget_under_mouse())
                # print(widget)
            #except:
            #    widget=''

            # if self.is_declared:
            #     get_text_first= self.first_box.get("1.0", "end-1c")
            #     get_text_second= self.second_box.get("1.0", "end-1c")

            #     if len(get_text_first) == 0 and widget!='.!frame.!frame2.!frame2.!text':
            #         self.first_box.delete("1.0",END)
            #         self.first_box.insert("1.0", "Paste selected sentence here...")
            #     elif get_text_first.replace('\n','')=="Paste selected sentence here..." and widget=='.!frame.!frame2.!frame2.!text':
            #         self.first_box.delete("1.0",END)
            #     elif get_text_first.replace('\n','')!="Paste selected sentence here..." and get_text_first.replace('\n','').find("Paste selected sentence here...")>=0:
            #         self.first_box.delete("1.0",END)
            #         self.first_box.insert("1.0",get_text_first.replace('\n','').replace("Paste selected sentence here...",''))


            #     if len(get_text_second) == 0 and widget!='.!frame.!frame2.!frame3.!text':
            #         self.first_box.delete("1.0",END)
            #         self.second_box.insert("1.0", "Paste selected sentence here...")
            #     elif get_text_second.replace('\n','')=="Paste selected sentence here..." and widget=='.!frame.!frame2.!frame3.!text':
            #         self.second_box.delete("1.0",END)
            #     elif get_text_second.replace('\n','')!="Paste selected sentence here..." and get_text_second.replace('\n','').find("Paste selected sentence here...")>=0:
            #         self.second_box.delete("1.0",END)
            #         self.second_box.insert("1.0",get_text_second.replace('\n','').replace("Paste selected sentence here...",''))

            if self.current_patent is not None:
                self.open_btn.pack(side=TOP,fill=BOTH,expand=True)
                # self.copy_btn.grid(row=17,column=5,columnspan=2)
            else:
                self.open_btn.grid_forget()
                # self.copy_btn.grid_forget()

            if self._thread is not None:
                if not self._thread.is_alive():
                    self._thread = None
                    self.current_action = None
                    self.update_label()

            if not self.outQueue1.empty():
                print("Getting the results")
                out1 = self.outQueue1.get()
                # print(out1[0])
                self.first_sentences_probabilities = out1[0][0]['first part']
                self.second_sentences_probabilities = out1[0][0]['second part']
                self.summary = out1[1][0]
                if self.predict_contradiction:
                    self.update_proba_label(out1[2][0][1])
                self._thread4 = Thread(target=self.find_sentences,args=(self.summary,self.area.get("1.0",END)))
                self._thread4.daemon = True
                self._thread4.start()


            if self.valid or self.scale_first.get()!=self.scale_value_first or self.scale_second.get()!=self.scale_value_second:
                self.valid=False
                self.scale_value_first, self.scale_value_second=self.scale_first.get(),self.scale_second.get()

                list_tags=[]
                for index in self.indexes_list_second[:self.scale_value_second]:
                    if index in self.indexes_list_first[:self.scale_value_first]:
                        list_tags.append('both_sentences')
                    else:
                        list_tags.append('second_sentences')

                self.area.tag_remove('first_sentences', 1.0, 'end')
                self.area.tag_remove('second_sentences', 1.0, 'end')
                self.area.tag_remove('both_sentences', 1.0, 'end')

                for idx,pos in self.indexes_list_first[:self.scale_value_first]:
                    self.area.tag_add('first_sentences', idx, pos)
                for i,(idx,pos) in enumerate(self.indexes_list_second[:self.scale_value_second]):
                    self.area.tag_add(list_tags[i], idx, pos)

                
                

                        

            time.sleep(0.05)

            
    def find_sentences(self,summary,area):        
        first_selected_text = summary['first part'][:self.max_scale]
        second_selected_text= summary['second part'][:self.max_scale]

        src = [area.split('.')[sentence].split(' ')[1:] for sentence in range(len(area.split('.')))]

        tgt_first= [first_selected_text[sentence].split(' ')[1:] for sentence in range(len(first_selected_text))]
        self.oracle_ids_first = greedy_selection(src, tgt_first, len(first_selected_text))

        tgt_second= [second_selected_text[sentence].split(' ')[1:] for sentence in range(len(second_selected_text))]
        self.oracle_ids_second = greedy_selection(src, tgt_second, len(second_selected_text))

        src2 = [self.area.get("1.0",END).split('.')[sentence] for sentence in range(len(self.area.get("1.0",END).split('.')))]
        self.indexes_list_first=[]
        self.indexes_list_second=[]
        for i in self.oracle_ids_first:
            self.indexes_list_first.append(self.search(src2[i], len(src2[i])))
        for i in self.oracle_ids_second:
            self.indexes_list_second.append(self.search(src2[i], len(src2[i])))

        self.valid = True
        print("Sentences found.")

    
    def analyze(self):
        
        if self._thread is None:

            #if already analysed
            if self.first_sentences_probabilities is None:
                if len(self.list_patents)>0:
                    if self.current_patent is not None:
                        # if os.path.isdir('./data/Preprocessing_results/'):
                        #     shutil.rmtree('./data/Preprocessing_results/')

                        # # Preprocessing
                        # call_extractor_preprocessor(task = 'Preprocessing', path='/'.join(self.current_patent.split('/')[:-1]))

                        # # Inference in separated threads
                        # path_data = './data/Preprocessing_results/Preprocessing_results_bert/STATE_OF_THE_ART'
                        t_analysis=Thread(target=apply_model, args=(self.current_patent,self.outQueue1,self.predict_contradiction))
                        t_analysis.start()
                        
                        # t_second=Thread(target=analyse_patent, args=(path_second_model,path_data,self.outQueue2,addon))
                        # t_second.start()

                        self._thread = t_analysis
                        self.current_action = 'Computation'
                        self.indicate_thread()

                    else:
                        self.choose_patent_message()
                else:
                    self.need_extraction_message()
            else:
                self.already_did_this_message()
        else:
            self.error_message()


    def search(self,start_exp, length_sentence):
        pos = '1.0'
        while True:
            print("searching...")
            idx = self.area.search(start_exp, pos, END)
            print("found")
            if not idx:
                break
            pos = '{}+{}c'.format(idx, length_sentence)
            return idx,pos
            #self.area.tag_add(tag, idx, pos)



    def add_sentence_first(self):
        if len(self.first_box.get("1.0", "end-1c")) != 0 and self.first_box.get("1.0",END).replace('\n','') != ("Paste selected sentence here..."):
            sentence = self.first_box.get("1.0",END).replace('\n','').replace('"Paste selected sentence here..."','')
            self.box_first.insert(END, sentence)
            self.first_box.delete("1.0",END)
            self.update_summary()

    def delete_sentence_first(self):
        if self.box_first.curselection() is not ():
            for index in self.box_first.curselection():
                item = self.box_first.get(int(index))
                self.box_first.delete(int(index))
                self.update_summary()

    def edit_sentence_first(self):
        if self.box_first.curselection() is not ():
            index= self.box_first.curselection()[0]
            item = self.box_first.get(int(index))
            self.first_box.delete("1.0",END)
            self.first_box.insert(END, self.box_first.get(int(index)))
            self.box_first.delete(int(index))
            self.update_summary()


    def add_sentence_second(self):
        if len(self.second_box.get("1.0", "end-1c")) != 0 and self.second_box.get("1.0",END).replace('\n','') != ("Paste selected sentence here..."):
            sentence = self.second_box.get("1.0",END).replace('\n','').replace('"Paste selected sentence here..."','')
            self.box_second.insert(END, sentence)
            self.second_box.delete("1.0",END)
            self.update_summary()

    def delete_sentence_second(self):
        if self.box_second.curselection() is not ():
            for index in self.box_second.curselection():
                item = self.box_second.get(int(index))
                self.box_second.delete(int(index))
                self.update_summary()

    def edit_sentence_second(self):
        if self.box_second.curselection() is not ():
            index = self.box_second.curselection()[0]
            item = self.box_second.get(int(index))
            self.second_box.delete("1.0",END)
            self.second_box.insert(END, self.box_second.get(int(index)))
            self.box_second.delete(int(index))
            self.update_summary()


    def copy_patent_directory(self):
        pass

    def update_summary(self):
        if self._thread is None:
            if self.current_patent is not None:
                with open('.'.join(self.current_patent.split('.')[:-1])+'.SUM','w') as writer_summary:
                    list_first = self.box_first.get(0,END)
                    list_second = self.box_second.get(0,END)
                    if list_first is not () and list_second is not ():
                        writer_summary.write('//\n'.join(list_first)+'///\n'+'//\n'.join(list_second))
                    elif list_first is not ():
                        writer_summary.write('\n//'.join(list_first))


    def read_existing_summary(self,path):
        with open('.'.join(path.split('.')[:-1])+'.SUM','r') as reader_summary:
            content = reader_summary.read()
            self.box_second.delete(0,END)
            self.box_first.delete(0,END)
            parts = content.split('///')
            if len(parts)>=1:
                print("ok1")
                for elmt in parts[0].split('//'):
                    if elmt is not '\n' and elmt is not '':
                        self.box_first.insert(END, elmt.replace('\n','').replace('STATE_OF_THE_ART',''))
                        if re.search(r"\s", elmt[0]):
                            idx,pos = self.search(elmt[1:40], len(elmt))
                            self.area.tag_add('first_sentences', idx, pos)
                        else:
                            idx,pos = self.search(elmt[0:40], len(elmt))
                            self.area.tag_add('first_sentences', idx, pos)
            if len(parts)==2:
                print("ok")
                for elmt in parts[1].split('//'):
                    if elmt is not '\n' and elmt is not '':
                        self.box_second.insert(END, elmt.replace('\n',''))
                        if re.search(r"\s", elmt[0]):
                            idx,pos = self.search(elmt[1:40], len(elmt))
                            self.area.tag_add('second_sentences', idx, pos)
                        else:
                            idx,pos = self.search(elmt[0:40], len(elmt))
                            self.area.tag_add('second_sentences', idx, pos)

    def print_widget_under_mouse(self):
        x,y = self.winfo_pointerxy()
        return self.winfo_containing(x,y)
        
        # self.after(1000, print_widget_under_mouse, self)


root = App()
root.tk.call('tk', 'scaling', 2.0)
root.mainloop()
