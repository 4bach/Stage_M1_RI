import numpy as np
import matchzoo 
import gensim
from bs4 import BeautifulSoup
import ast
from os import listdir
from os.path import isfile, join



def get_all_docs_robust4(folder="/local/karmim/Stage_M1_RI/data/collection"):
    dossier = ['FBIS','FR94','FT','LATIMES']
    all_doc ={}

    for d in dossier : 
        all_doc[d] = [f for f in listdir(folder+'/'+d) if isfile(join(folder+'/'+d, f))]





class DRMM_Matchzoo:

    def __init__(self,all_doc):
        """

        """
        self.d_query = {}
        self.all_doc= all_doc



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
        with open(file_doc,"r") as f:
        soup = BeautifulSoup(f.read(),"xml")
        #stemmer = SnowballStemmer("french", ignore_stopwords=True)        
        
        return np.array([ d.text for d in soup.find_all("TEXT")])

