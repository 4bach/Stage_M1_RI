import numpy as np
#import matchzoo 
import gensim
import random
import ast
import json
import os
import pickle
import time
import progressbar
import codecs
from gensim.models import KeyedVectors
from bs4 import BeautifulSoup
from os import listdir,sep
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer
import fasttext # On utilise fastText car il fait automatiquement le prétraitement pour les mots inconnus. 
from gensim.models.wrappers import FastText






class Dataset:

    def __init__(self,intervals,normalize,CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords,strip_numeric, strip_tags, strip_punctuation] ,embeddings_path="/local/karmim/Stage_M1_RI/data/vocab"):
        """
            all_doc : dictionnaire de tout nos documents afin d'itérer dessus. 

        """
        self.d_query = {} # Notre dictionnaire de query
        self.CUSTOM_FILTERS = CUSTOM_FILTERS # Liste de fonction de Préprocessing des docs
        self.intervals = intervals
        self.intvlsArray = np.linspace(-1, 1, self.intervals)
        self.max_length_query = 0
        self.docs = {}
        self.normalize = normalize
        self.model_wv = FastText.load_fasttext_format(embeddings_path + sep + "parameters.bin")

    def set_params(self, vectoriser, idf_file="idf_robust2004.pkl",robust_path="/local/karmim/Stage_M1_RI/data/collection"):
        
        self.vectoriser = vectoriser 
        #dict: term -> idf
        self.idf_values = pickle.load(open(idf_file, "rb"))
        self.robust_path = robust_path

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

    def load_all_query(self,file_query="/local/karmim/Stage_M1_RI/data/robust2004.txt"):
        """
            On recupère toutes les querys qui sont ensuite sauvegardées dans un dictionnaire. 

        """
        f = open(file_query,"r")
        self.d_query = ast.literal_eval(f.read())
        f.close()
        for k in self.d_query :
            self.d_query[k]= self.d_query[k][0].split(' ') # On suppr les query langage naturel, et on met la query mot clé sous forme de liste.
        self.max_length_query =  np.max([len(self.d_query[q]) for q in self.d_query])
        print("query chargé\n")

    def load_doc(self,file_doc,pre_process=True):
        """
            Fonction qui load un fichier file_doc. 
            pre_process -> Bool qui dit si on effectue le preprocessing ou non. 
        """
        cpt_err = 0
        with codecs.open(file_doc,'r',encoding='utf-8',errors='ignore') as f:
            soup = BeautifulSoup(f.read(),"html.parser")
        id_ = soup.find_all('docno')
        text_ = soup.find_all('text')
        
        for i in range(len(id_)):
            try:
                self.docs[(id_[i].text).strip()] =  preprocess_string(text_[i].text, self.CUSTOM_FILTERS)[2:]
            except IndexError:
                cpt_err+=1
                print("len des id: ",len(id_)," i courant : ",i, " erreur: ",cpt_err)
        return self.docs


    def load_all_docs(self,doc_json="../data/object_python/all_docs_preprocess.json"):
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
    
    def load_docs_per_folder(self,path_json = '/local/karmim/Stage_M1_RI/data/object_python/',path_collection="/local/karmim/Stage_M1_RI/data/collection"):
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

    def load_relevance(self,file_rel="/local/karmim/Stage_M1_RI/data/qrels.robust2004.txt"):
        """
            Chargement du fichier des pertinences pour les requêtes. 
            Pour chaque paire query/doc on nous dit si pertinent ou non. 
        """
        self.paires = {}
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
        
        print("relevance chargé\n")
        return self.paires


    def hist(self, query, document):
        """
        """
        X = []
        for i in query.nonzero()[1]:
            histo = []
            for j in document.nonzero()[1]:
                histo.append(cosine_similarity([self.model_wv.vectors[i]], [self.model_wv.vectors[j]])[0][0])
            histo, _ = np.histogram(histo, bins=self.intvlsArray)
            if self.normalize:
                histo = histo / histo.sum()
            X.append(histo)
        if len(query.nonzero()[1]) < self.max_length_query:
            # compléter avec des zéro
            for i in range(self.max_length_query - len(query.nonzero()[1])):
                X.append([0]*self.intervals)
        #retourner histogramme interactions
        return np.array(X)

    def prepare_data_forNN(self, test_size=0.2):
        """
        """

        #spliter les requêtes en train/test
        lol = [q for q in self.d_query.keys() if q in self.paires]
        random.shuffle(lol)
        test_keys = lol[:int(test_size * len(lol))]
        train_keys = lol[int(test_size * len(lol)):]
        
        #pour chaque requête on va générer autant de paires relevant que irrelevant
        #pour nos besoins on va alterner paires positives et paires négatives
        train_hist = [] # les histogrammes d'interraction
        test_hist = []
        train_idf = [] #les vecteurs d'idf
        test_idf = []
        
        for id_requete in train_keys:
            #recuperer les mots dont on connait les embeddings dans la query
            q = self.vectoriser.transform([' '.join(self.d_query[id_requete])])
            idf_vec = self.get_idf_vec(self.d_query[id_requete])
            for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
                #lire le doc, la requete et creer l'histogramme d'interraction
                
                d = self.vectoriser.transform([' '.join(self.docs[pos])])
                train_hist.append(self.hist(q, d)) #append le doc positif
                train_idf.append(idf_vec) #append le vecteur idf de la requête
                
                
                d = self.vectoriser.transform([' '.join(self.docs[neg])])
                train_hist.append(self.hist(q, d)) #append le doc négatif
                train_idf.append(idf_vec) #append le vecteur idf de la requête
        train_labels = np.zeros(len(train_hist))
        train_labels[::2] = 1 # label de pertinence 
        
        
        for id_requete in test_keys:
            #recuperer les mots dont on connait les embeddings dans la query
            q = self.vectoriser.transform([' '.join(self.d_query[id_requete])])
            idf_vec = self.get_idf_vec(self.d_query[id_requete])
            for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
                #lire le doc, la requete et creer l'histogramme d'interraction
                
                d = self.vectoriser.transform([' '.join(self.docs[pos])])

                test_hist.append(self.hist(q, d)) #append le doc positif
                test_idf.append(idf_vec) #append le vecteur idf de la requête
                
                d = self.vectoriser.transform([' '.join(self.docs[neg])])


                test_hist.append(self.hist(q, d)) #append le doc négatif
                test_idf.append(idf_vec) #append le vecteur idf de la requête
        test_labels = np.zeros(len(train_hist))
        test_labels[::2] = 1
        
        return (train_hist, train_idf, train_labels), (test_hist, test_idf, test_labels)
        
        #éventuellement sauvegarder tout ça sur le disque comme ça c fait une bonne fois pour toutes...