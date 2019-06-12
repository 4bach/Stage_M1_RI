import numpy as np
import matchzoo 
import gensim
import ast
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
        self.vectorizer = CountVectorizer(analyzer='word', vocabulary=vocabulary, binary=True)




    def load_all_query(self,file_query="/local/karmim/Stage_M1_RI/data/robust2004.txt"):
        """
            On recupère toutes les querys qui sont ensuite sauvegardées dans un dictionnaire. 

        """
        f = open(file_query,"r")
        self.d_query = ast.literal_eval(f.read())
        f.close()
        for k in self.d_query :
            self.d_query[k]= self.d_query[k][0].split(' ') # On suppr les query langage naturel, et on met la query mot clé sous forme de liste.

    
    def load_docs(self,file_doc):
        self.current_docs={}
        with open(file_doc,"r") as f:
            soup = BeautifulSoup(f.read(),"html.parser")
        id_ = soup.find_all('docno')
        text_ = soup.find_all('text')
        
        for i in range(len(id_)):
            self.current_docs[id_[i].text] =  preprocess_string(text_[i].text, self.CUSTOM_FILTERS)[2:]


    def embedding_query(self):
        pass
    

