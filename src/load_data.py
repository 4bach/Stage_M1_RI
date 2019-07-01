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
import progressbar
import codecs
from gensim.models import KeyedVectors
from bs4 import BeautifulSoup
from os import listdir,sep
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation
#from sklearn.feature_extraction.text import CountVectorizer
#import fasttext # On utilise fastText car il fait automatiquement le prétraitement pour les mots inconnus. 
from gensim.models.wrappers import FastText






class Dataset:

    def __init__(self,CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords,strip_numeric, strip_tags, strip_punctuation] ,embeddings_path="/local/karmim/Stage_M1_RI/data/vocab"):
        """
            all_doc : dictionnaire de tout nos documents afin d'itérer dessus. 

        """
        self.d_query = {} # Notre dictionnaire de query
        self.CUSTOM_FILTERS = CUSTOM_FILTERS # Liste de fonction de Préprocessing des docs
        #self.intervals = intervals
        #self.intvlsArray = np.linspace(-1, 1, self.intervals)
        self.max_length_query = 0
        self.docs = {} # Dico de tout les documents de robust4
        #self.normalize = normalize
        self.model_wv = FastText.load_fasttext_format(embeddings_path + sep + "parameters.bin")

    def set_params(self, idf_file="idf_robust2004.pkl",robust_path="/local/karmim/Stage_M1_RI/data/collection"):
        
        # self.vectoriser = vectoriser 
        #dict: term -> idf
        self.idf_values = pickle.load(open(idf_file, "rb"))
        self.robust_path = robust_path
        self.embedding_query()
        self.embedding_doc()
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
            self.max_length_query =  np.max([len(self.d_query[q]) for q in self.d_query])
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
                    if l[-1]=='1':
                        self.paires[l[0]]['relevant'].append(l[2])
                    else:
                        self.paires[l[0]]['irrelevant'].append(l[2])
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
        self.query_emb = self.d_query.copy()
        cpt=0
        if not exists : 
            for k in self.d_query: 
                for i,w in enumerate(self.d_query[k]): 
                #print(w) 
                #print(type(w)) 
                 if  w in self.model_wv: 
                    self.query_emb[k][i] = self.model_wv[w]
                    
                else:
                    cpt+=1
            print("Nombre de mots ignorés :",cpt)
            pickle.dump( self.query_emb, open( file_pkl, "wb" ) )
            print("Le fichier emb_query.pkl a bien été enregistré.")
        
        else:    
            print("Chargement du fichier pickle : emb_query.pkl ...")
            self.query_emb = pickle.load( open( file_pkl, "rb" ) )
            print("Chargement emb_query.pkl réussi")

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
            pickle.dump( self.docs_emb, open( file_pkl, "wb" ) )
            print("Le fichier emb_docs.pkl a bien été enregistré.")
        
        else:    
            print("Chargement du fichier pickle : emb_docs.pkl ...")
            self.docs_emb = pickle.load( open( file_pkl, "rb" ) )
            print("Chargement emb_docs.pkl réussi")

        return self.docs_emb
