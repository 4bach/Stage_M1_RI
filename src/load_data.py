import numpy as np 
#import matchzoo 
import gensim
import random
import ast
import json
import os
import pickle
import time
import torch
import copy
import sys
import progressbar
import codecs
from gensim.models import KeyedVectors
from bs4 import BeautifulSoup
from os import listdir,sep
import gensim.models.word2vec as w2v 
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation
#from sklearn.feature_extraction.text import CountVectorizer
#import fasttext # On utilise fastText car il fait automatiquement le prétraitement pour les mots inconnus. 
from gensim.models.wrappers import FastText
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
if not sys.warnoptions:
    warnings.simplefilter("ignore")






class Dataset:

    def __init__(self,load_all=False,w2v_model=False,CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords,strip_numeric, strip_tags, strip_punctuation] ,embeddings_path="/local/karmim/Stage_M1_RI/data/vocab"):
        """
            w2v_model : If use a word2vec model with gensim True else False 
            CUSTOM_FILTERS : list of filters use for the preprocessing 
            embedding_path : path to the embedding model to load. 
            

            CAREFUL -> CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords] use this filter if u load a w2v model.
        """
        self.d_query = {} # Notre dictionnaire de query
        self.CUSTOM_FILTERS = CUSTOM_FILTERS # Liste de fonction de Préprocessing des docs
        #self.intervals = intervals
        #self.intvlsArray = np.linspace(-1, 1, self.intervals)
        self.max_length_query = 0
        self.docs = {} # Dico de tout les documents de robust4
        #self.normalize = normalize
        if not w2v_model:
            self.model_wv = FastText.load_fasttext_format(embeddings_path + sep + "parameters.bin")
        else:
            
            # embeddings_path for w2v can be 
            #   /local/karmim/Stage_M1_RI/data/object_python/concept_part/modelw2v/no_low_frequencyRobustConcept2v.w2v
            #   /local/karmim/Stage_M1_RI/data/object_python/concept_part/modelw2v/allRobustConcept2v.w2v
            self.model_wv =  w2v.Word2Vec.load(embeddings_path)

        if load_all:
            print("load all necessary object...")
            self.load_relevance()
            self.load_all_docs()
            self.embedding_query()
            #self.load_all_query()
            self.max_length_query = np.max([len(self.query_emb[q]) for q in self.query_emb])

    def set_params(self, idf_file="idf_robust2004.pkl",robust_path="/local/karmim/Stage_M1_RI/data/collection"):
        
        # self.vectoriser = vectoriser 
        #dict: term -> idf
        self.idf_values = pickle.load(open(idf_file, "rb"))
        self.robust_path = robust_path
        self.embedding_query()
        
    def get_vocab(self):

        return self.model_wv.wv.vocab

    def get_query(self,key,affiche=True):
        if affiche:
            print("query: ",key," ",self.d_query[key])
        return self.d_query[key]

    def get_doc(self,key,affiche=True):
        if affiche:
            print("doc: ",key," ",self.docs[key])
        return self.docs[key]

    def get_relevance(self,q_id,affiche=True):
        if affiche: 
            print("query: ",q_id," ",self.paires[q_id]['relevant'])
        return self.paires[q_id]['relevant']

    def get_idf_vec(self,id_query):
        """
        """
        vec = np.zeros(self.max_length_query)
        for i,term in enumerate(self.d_query[id_query]) :
            if term in self.model_wv: #je suis pas sûr de mon coup là
                if term in self.idf_values:
                    vec[i] = self.idf_values[term]
                elif term.lower() in self.idf_values:
                    vec[i] = self.idf_values[term.lower()]
                else:
                    vec[i] = 1.
                
        return vec

    def load_all_path_docs_robust4(self,folder="/local/karmim/Stage_M1_RI/data/collection"):
        """
            On charge tout les path des fichiers de la collection robust 4 dans une liste 
        """
        
        self.all_doc =[]
        for r,_,f in os.walk(folder): 
            for file in f: 
                self.all_doc.append(os.path.join(r,file)) 

    def load_all_query(self,file_query="/local/karmim/Stage_M1_RI/data/robust2004.txt",file_json="/local/karmim/Stage_M1_RI/data/object_python/query.json"):

        """
            On recupère toutes les querys qui sont ensuite sauvegardées dans un dictionnaire. 

        """
        self.d_query={}
        exists = os.path.isfile(file_json)
        if not exists : 
            f = open(file_query,"r")
            self.d_query = ast.literal_eval(f.read())
            f.close()
            for k in self.d_query :
                self.d_query[k]= self.d_query[k][0].split(' ') # On suppr les query langage naturel, et on met la query mot clé sous forme de liste.
            
            print("query chargé\n")
            save = json.dumps(self.d_query)
            f = open(file_json,"w")
            f.write(save)
            f.close()
            print("query.json save")
        else:
            print("Chargement du fichier query.json")
            with open(file_json) as json_file:
                self.d_query = json.load(json_file)
            print("query chargé\n")
        self.max_length_query =  np.max([len(self.d_query[q]) for q in self.d_query])
        print("Longueur max query : ",self.max_length_query)
        return self.d_query
            
    def load_relevance(self,file_rel="/local/karmim/Stage_M1_RI/data/qrels.robust2004.txt",file_json="/local/karmim/Stage_M1_RI/data/object_python/qrel.json"):
        
        """
            Chargement du fichier des pertinences pour les requêtes. 
            Pour chaque paire query/doc on nous dit si pertinent ou non. 
        """
        self.paires = {}
        exists = os.path.isfile(file_json)
        if not exists:
            with open(file_rel,"r") as f:
                for line in f :
                    l = line.split(' ')
                    self.paires.setdefault(l[0],{})
                    self.paires[l[0]].setdefault('relevant',[])
                    self.paires[l[0]].setdefault('irrelevant',[])
                    if l[-1].rstrip()=='0':
                        self.paires[l[0]]['irrelevant'].append(l[2])
                    else:
                        self.paires[l[0]]['relevant'].append(l[2])
            save = json.dumps(self.paires)
            f = open(file_json,"w")
            f.write(save)
            f.close()
            print("Le fichier qrel.json a bien été enregistré.")

        else:
            print("Chargement du fichier qrel.json")
            with open(file_json) as json_file:
                self.paires = json.load(json_file)
            print("relevance chargé\n")
        return self.paires
        
        


    def load_doc(self,file_doc,pre_process=True):
        """
            Fonction qui load un fichier file_doc. 
            pre_process -> Bool qui dit si on effectue le preprocessing ou non. 
        """
        
        with codecs.open(file_doc,'r',encoding='utf-8',errors='ignore') as f_:
            soup = BeautifulSoup(f_.read(),"html.parser")
        docs = soup.find_all('doc')
        for d_ in docs :  
            self.docs[d_.docno.text.strip()]=preprocess_string(d_.text,self.CUSTOM_FILTERS)
        
        return self.docs


    def load_all_docs(self,doc_json="/local/karmim/Stage_M1_RI/data/object_python/all_docs_preprocess.json"):
        """
            Charge tout les docs dans un dico, puis les enregistre dans un fichier json. 
            Charge directement le fichier doc_json s'il existe.  
        """
        self.load_all_path_docs_robust4()
        exists = os.path.isfile(doc_json)
        if not exists:
            for f in self.all_doc:
                
                print("f -> ",f)
                self.load_doc(f)

            save = json.dumps(self.docs)
            f = open(doc_json,"w")
            f.write(save)
            f.close()

        else:
            print("Chargement du fichier json : ",doc_json," ...")
            with open(doc_json) as json_file:
                self.docs = json.load(json_file)
                
        print("docs chargé\n")

        return self.docs

    def load_docs_per_folder(self,path_json = '/local/karmim/Stage_M1_RI/data/object_python/',path_collection="/local/karmim/Stage_M1_RI/data/collection"):
        
        """
            Fonction qui charge tout les docs mais en construisant 4 dictionnaires des 4 dossier FBIS,FR94,FT,LATIMES.
        """
        folder = ['FBIS','FR94','FT','LATIMES']
        self.fbis = {}
        self.fr94 = {}
        self.ft = {}
        self.latimes = {}
        l = [self.fbis,self.fr94,self.ft,self.latimes]
        for i,d in enumerate(folder) : 
            path=path_collection+'/'+d

            self.load_all_path_docs_robust4()
            exists = os.path.isfile(path_json+d+'.json')
            if not exists:
                for r,_,f in os.walk(path): 
                    for files in f: 
                        fi = os.path.join(r,files)
                        
                        with codecs.open(fi,'r',encoding='utf-8',errors='ignore') as f_:
                            soup = BeautifulSoup(f_.read(),"html.parser")
                        docs = soup.find_all('doc')
                        for d_ in docs :  
                            l[i][d_.docno.text.strip()]=preprocess_string(d_.text,self.CUSTOM_FILTERS)
                        
                save = json.dumps(l[i])
                f = open(path_json+d+'.json',"w")
                f.write(save)
                f.close()
                print("Le fichier "+path_json+d+'.json a bien été enregistré.')

            else:
                print("Chargement du fichier json : "+path_json+d+'.json ...')
                with open(path_json+d+'.json') as json_file:
                    l[i] = json.load(json_file)
        return (self.fbis,self.fr94,self.ft,self.latimes)

    
    def embedding_query(self,file_pkl="/local/karmim/Stage_M1_RI/data/object_python/emb_query.pkl" ):
        """
            Fonction qui transforme nos mots du dictionnaire de query par des embeddings (vecteurs).
        """
        exists = os.path.isfile(file_pkl)
        self.query_emb = copy.deepcopy(self.d_query)
        cpt=0
        if not exists : 
            for k in self.d_query:
                self.query_emb[k] = [] 
                for w in self.d_query[k]:  
                    if  w in self.model_wv: 
                        self.query_emb[k].append(self.model_wv[w])
                        
                    else:
                        
                        cpt+=1
            print("Nombre de mots ignorés :",cpt)
            pickle.dump( self.query_emb, open( file_pkl, "wb" ) )
            print("Le fichier emb_query.pkl a bien été enregistré.")
        
        else:    
            print("Chargement du fichier pickle : emb_query.pkl ...")
            self.query_emb = pickle.load( open( file_pkl, "rb" ) )
            print("Chargement emb_query.pkl réussi")
            
        self.max_length_query = np.max([len(self.query_emb[q]) for q in self.query_emb])
        return self.query_emb

    def embedding_doc(self,file_pkl="/local/karmim/Stage_M1_RI/data/object_python/emb_docs.pkl"):
        """
            Prends trop de mémoire. 
            Chargement direct des interractions.
        """
        exists = os.path.isfile(file_pkl)
        self.docs_emb = self.docs.copy()
        cpt=0
        if not exists : 
            for k in self.docs: 
                for i,w in enumerate(self.docs[k]): 
                #print(w) 
                #print(type(w)) 
                 if  w in self.model_wv: 
                    self.docs_emb[k][i] = self.model_wv[w]
                    
                else:
                    cpt+=1
            print("Nombre de mots ignorés :",cpt)
            with open(file_pkl,"wb") as f :

                pickle.dump( self.docs_emb, f )
            print("Le fichier emb_docs.pkl a bien été enregistré.")
        
        else:
            with open(file_pkl,"wb") as f :
                print("Chargement du fichier pickle : emb_docs.pkl ...")
                self.docs_emb = pickle.load( f )
            print("Chargement emb_docs.pkl réussi")

        return self.docs_emb



    def histo(self, query_id,doc_id,intervals=30,histo_type='CH'):
        """
            for a query embedded, a doc_id and the data object return the interaction histogram matrix. 
                intervals -> number of bins in a histogram
                histo_type -> The three types of histograms in DRMM original architecure : CH, NH, LCH.

        """

        #embedded docs 
        doc_emb = []
        cpt = 0
        for i,w in enumerate(self.docs[doc_id]): 
            #print(w) 
            #print(type(w)) 
            if  w in self.model_wv: 
                doc_emb.append( self.model_wv[w] )                 
            elif w.upper() in self.model_wv:
                doc_emb.append( self.model_wv[w.upper()] )
            elif w.lower() in self.model_wv:
                doc_emb.append(self.model_wv[w.lower()])
            else : 
                cpt+=1

        doc_emb = np.array(doc_emb)
        query = self.query_emb[query_id]
        if histo_type=='LCH':
            try:
                cos = cosine_similarity(query,doc_emb)
                mat_hist = np.log([np.histogram(cos[j],bins=intervals,range=(-1.0,1.0))[0] if (j < len(query)) \
                     else np.zeros((intervals,)) for j in range(self.max_length_query)])
                mat_hist[mat_hist < 0] = 0
            except ValueError:
                print("query :",query_id)
                print("doc :",doc_emb)

        else:
            try:
                cos = cosine_similarity(query,doc_emb)
                mat_hist = np.array([np.histogram(cos[j],bins=intervals,range=(-1.0,1.0))[0] if j < len(query) \
                    else np.zeros((intervals,)) for j in range(self.max_length_query)]) 
                
                
            except ValueError:
                print("query :",query)
                print("doc :",doc_emb)
            

            if histo_type == 'NH':
                mat_hist = np.array([i/i.sum() if i.sum()!= 0 else np.zeros(np.shape(i)) for i in mat_hist])
            
        return mat_hist 




    def calcul_all_interaction_forNN(self,intervals = 30,histo_type='CH',train_size=0.8,folder_interaction ="/local/karmim/Stage_M1_RI/data/object_python/interaction/no_concept/"):
        """
            This function compute all the interaction needed to train our Neural Network.  
                Input -> 
                    data : object from the file load_data.py 
                    intervals : number of bins in the histogram 
                    histo_type : The type of the histogram : CH, NH or LCH.
                    train_size : % of query that are use for the train   
                Output -> List of interaction matrix, with pos and neg examples. 
        """
        
        relevance = self.load_relevance()
        self.load_all_docs()
        que = np.array(list(relevance.keys())) # Split de nos query en train / test 
        random.shuffle(que)
        r = int(train_size*len(que))
        train_q = que[:r] # List of training queries id 
        test_q = que[r:]  # List of testing queries id
        train_X = {}   # List of all interaxion matrix from the train
        test_X = {}    # List of all interaxion matrix from the test
        #y_train = np.array([1 if i%2==0 else 0 for i in len(train_q)]) # We need an ordonate list of relevant irrelevant document...
        #y_test = np.array([1 if i%2==0 else 0 for i in len(test_q)])
        # train_X[query][0] = all the pos interaction train_X[query][1] all the neg interaxion...
        for id_q in train_q:
            exists = os.path.isfile(folder_interaction+id_q+histo_type+"_interractions.npy")
            if not exists : 
                all_interaction_4_query = []
                pos = np.array([self.histo(id_q,doc_pos,intervals=intervals,histo_type=histo_type) for doc_pos in relevance[id_q]['relevant']])
                all_interaction_4_query.append(pos)
                neg = np.array([self.histo(id_q,doc_neg,intervals=intervals,histo_type=histo_type) for doc_neg in relevance[id_q]['irrelevant']])
                all_interaction_4_query.append(neg)
                all_interaction_4_query = np.array(all_interaction_4_query)
                train_X[id_q] = all_interaction_4_query
                np.save(folder_interaction+id_q+histo_type+"_interractions.npy", all_interaction_4_query)
                print(folder_interaction+id_q+histo_type+"_interractions.npy succesfully saved.")
            else:
                all_interaction_4_query = np.load(folder_interaction+id_q+histo_type+"_interractions.npy",allow_pickle=True)
                train_X[id_q] = all_interaction_4_query
                print(folder_interaction+id_q+histo_type+"_interractions.npy succesfully loaded.")

        for id_q in test_q:
            exists = os.path.isfile(folder_interaction+id_q+histo_type+"_interractions.npy")
            if not exists : 
                all_interaction_4_query = []
                pos = np.array([self.histo(id_q,doc_pos,intervals=intervals,histo_type=histo_type) for doc_pos in relevance[id_q]['relevant']])
                all_interaction_4_query.append(pos)
                neg = np.array([self.histo(id_q,doc_neg,intervals=intervals,histo_type=histo_type) for doc_neg in relevance[id_q]['irrelevant']])
                all_interaction_4_query.append(neg)
                all_interaction_4_query = np.array(all_interaction_4_query)
                test_X[id_q] = all_interaction_4_query
                np.save(folder_interaction+id_q+histo_type+"_interractions.npy", all_interaction_4_query)
                print(folder_interaction+id_q+histo_type+"_interractions.npy succesfully saved.")
            else:
                all_interaction_4_query = np.load(folder_interaction+id_q+histo_type+"_interractions.npy",allow_pickle=True)
                test_X[id_q] = all_interaction_4_query
                print(folder_interaction+id_q+histo_type+"_interractions.npy succesfully loaded.")
        print("all interaction succesfully loaded/saved...")
        return train_X,test_X




    def get_interaction(self,id_q,histo_type,folder_interaction ="/local/karmim/Stage_M1_RI/data/object_python/interaction/no_concept/" ):


        all_interaction_4_query = np.load(folder_interaction+id_q+histo_type+"_interractions.npy",allow_pickle=True)
        return all_interaction_4_query