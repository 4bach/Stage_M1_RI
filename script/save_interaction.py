"""
    In this file we save and compute all the interactions from the file qrel.

"""
import sys
sys.path.insert(0, '/local/karmim/Stage_M1_RI/src')
import load_data
import numpy as np
import random 
from sklearn.metrics.pairwise import cosine_similarity
import time
import os 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation
if not sys.warnoptions:
    warnings.simplefilter("ignore")




if __name__ == "__main__":
    data = load_data.Dataset(load_all=False,w2v_model=True,CUSTOM_FILTERS= [lambda x: x.lower(),remove_stopwords],embeddings_path="/local/karmim/Stage_M1_RI/data/object_python/concept_part/modelw2v/allRobustConcept2v.w2v")
    #relevance = data.load_relevance()
    docs = data.load_all_docs(doc_json="/local/karmim/Stage_M1_RI/data/object_python/concept_part/preprocess_doc.json")
    #query = data.load_all_query()
    data.embedding_query(file_pkl="/local/karmim/Stage_M1_RI/data/object_python/concept_part/emb_query_allrobust.pkl")
    #max_length = data.max_length_query
    #cosine_similarity(data.model_wv['car'].reshape(1,-1),data.model_wv['truck'].reshape(1,-1)).item()
    #hist = np.array([cosine_similarity(que_emb['301'][0].reshape(1,-1),data.model_wv[i].reshape(1,-1)).item() for i in docs['LA070389-0001']])
    #ch = histo(que_emb['301'],'LA070389-0001',data)
    #lch = histo(que_emb['301'],'LA070389-0001',data,histo_type='LCH') # Work well for the DRMM architecture. 
    #nh = histo(que_emb['301'],'LA070389-0001',data,histo_type='NH') # Dont work really good in the original paper.
    data.calcul_all_interaction_forNN(intervals = 30,histo_type='LCH',train_size=0.8,folder_interaction ="/local/karmim/Stage_M1_RI/data/object_python/interaction/all_robust/")
    data.calcul_all_interaction_forNN(intervals = 30,histo_type='CH',train_size=0.8,folder_interaction ="/local/karmim/Stage_M1_RI/data/object_python/interaction/all_robust/")