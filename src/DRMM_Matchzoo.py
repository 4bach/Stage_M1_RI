import numpy as np
import matchzoo 
import gensim
import ast
import json
from gensim.models import KeyedVectors
from bs4 import BeautifulSoup
from os import listdir,sep
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation
from sklearn.feature_extraction.text import CountVectorizer


def get_all_docs_robust4(folder="/local/karmim/Stage_M1_RI/data/collection"):
    dossier = ['FBIS','FR94','FT','LATIMES']
    all_doc ={}

    for d in dossier : 
        all_doc[d] = [f for f in listdir(folder+'/'+d) if isfile(join(folder+'/'+d, f))]
    return all_doc




class DRMM_Matchzoo:

    def __init__(self,all_doc,CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords,strip_numeric, strip_tags, strip_punctuation],embeddings_path="/local/karmim/Stage_M1_RI/data/vocab" ):
        """
            all_doc : dictionnaire de tout nos documents afin d'itérer dessus. 

        """
        self.d_query = {} # Notre dictionnaire de query
        self.all_doc= all_doc # Liste de tout nos documents
        self.CUSTOM_FILTERS = CUSTOM_FILTERS # Liste de fonction de Préprocessing des docs
        self.model = KeyedVectors.load_word2vec_format(embeddings_path + sep + "model.bin", binary=True)
        self.vocabulary = [w for w in self.model.vocab]
        self.vectorizer = CountVectorizer(analyzer='word', vocabulary=self.vocabulary, binary=True)
        self.max_length_query = 0
        self.current_docs = {}
        self.json_doc_exist = False



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
    
    def load_doc(self,file_doc):
        with open(file_doc,"r") as f:
            soup = BeautifulSoup(f.read(),"html.parser")
        id_ = soup.find_all('docno')
        text_ = soup.find_all('text')
        
        for i in range(len(id_)):
            self.current_docs[id_[i].text] =  preprocess_string(text_[i].text, self.CUSTOM_FILTERS)[2:]
        
        return self.current_docs


    def load_all_docs(self,doc_json="../data/object_python/all_docs_preprocess.json"):
        """
            Charge tout les docs dans un dico. 
        """
        if not self.json_doc_exist:
            for i in ['FBIS','FR94','FT','LATIMES']:
                for doc in self.all_doc[i]:
                    self.load_doc(doc)

            save = json.dumps(self.current_docs)
            f = open(doc_json,"w")
            f.write(save)
            f.close()
            self.json_doc_exist = True

        else:
            self.current_docs = json.load(doc_json)

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
        return self.paires

    def embedding_dictionnary(self,dico):
        """
            Fonction qui pour un dictionnaire avec comme clé l'ID et valeur une liste de mot 
            retourne un dictionnaire avec une liste des représentations vectorielle des mots (embedding)
            en utilisant Word2Vec.
        """

        
        

    

