from __future__ import absolute_import, division, print_function
import codecs 
import glob 
import logging 
import multiprocessing
import os 
import pprint
import re
import ast
import json
from bs4 import BeautifulSoup
import nltk 
import gensim.models.word2vec as w2v 
import sklearn.manifold # dimensionality reduction for visualisation.
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation

def load_all_path_docs_robust4(folder="/local/karmim/Stage_M1_RI/data/annotated_collection_tagme_score/015"):
        """
            We load all the path of all the annotated document of the robust4 collection.
        """
        
        all_file_name =[]
        for r,_,f in os.walk(folder): 
            for file in f: 
                all_file_name.append(os.path.join(r,file))
        return all_file_name

all_file_name = load_all_path_docs_robust4()

def load_all_query_annotated_robust4(file = '/local/karmim/Stage_M1_RI/data/topics-title.annotated.csv'):
    query_an = {} # Dict with words and concept for a query id
    concept = {} # Dict with only the concept for a query id
    f = codecs.open(file,'r',encoding='utf-8',errors='ignore')
    for line in f: 
        #print(line.split())
        line = np.array(line.split())
        index = np.where(np.char.find(line, '$#')>=0)
        concept[line[0]] = line[index]
        query_an[line[0]] = line[1:]
    return query_an,concept

q,c = load_all_query_annotated_robust4()


def load_doc(file_doc,all_docs={},all_concept={},pre_process=True):
        """
            Fonction qui load un fichier file_doc. 
            pre_process -> Bool qui dit si on effectue le preprocessing ou non. 
        """
        
        with codecs.open(file_doc,'r',encoding='utf-8',errors='ignore') as f_:
            soup = BeautifulSoup(f_.read(),"html.parser")
        docs = soup.find_all('doc')
        for d_ in docs :
            text = np.array(d_.text.split()[1:])
            doc_id = d_.docno.text.strip()
            all_docs[doc_id] = list(text)
            index = np.where(np.char.find(text, '$#')>=0)
            all_concept[doc_id] = list(text[index])
        return all_docs,all_concept

def load_all_doc(all_file,doc_json="/local/karmim/Stage_M1_RI/data/object_python/concept_part/anotated_doc.json",concept_doc_json="/local/karmim/Stage_M1_RI/data/object_python/concept_part/all_concept_doc.json"):
    exists = os.path.isfile(doc_json)
    all_doc = {}
    all_concept={}
    if not exists:
        
        for f in all_file:

            print("f -> ",f)
            load_doc(f,all_doc,all_concept)

        save = json.dumps(all_doc)
        f = open(doc_json,"w")
        f.write(save)
        f.close()
        print("document annoté sauvegardé...")
        save = json.dumps(concept_doc_json)
        f = open(doc_json,"w")
        f.write(save)
        f.close()
        print(" concept des documents sauvegardé...")
    else:
        
        print("Chargement du fichier json : anotated_doc.json ...")
        with open(doc_json) as json_file:
            all_doc = json.load(json_file)
        print("Chargement du fichier json : all_concept_doc.json ...")
        with open(concept_doc_json) as json_file:
            all_concept = json.load(json_file)

    #
    return all_doc,all_concept


def pre_process_doc(all_document={},all_concept={},CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords],json_doc_preprocess="/local/karmim/Stage_M1_RI/data/object_python/concept_part/preprocess_doc.json",json_concept_preprocess="/local/karmim/Stage_M1_RI/data/object_python/concept_part/preprocess_concept.json"):
    exists1 = os.path.isfile(json_doc_preprocess)
    exists2 = os.path.isfile(json_concept_preprocess)
    if not exists1 or not exists2:
        for k in all_document : 
            all_document[k] = preprocess_string(' '.join(all_document[k]),CUSTOM_FILTERS)
            all_concept[k] = preprocess_string(' '.join(all_document[k]),[lambda x : x.lower()]) # We only need to low the concept
        
        save = json.dumps(all_document)
        f = open(json_doc_preprocess,"w")
        f.write(save)
        f.close()
        print("document annoté preprocessé sauvegardé...")
        save = json.dumps(all_concept)
        f = open(json_concept_preprocess,"w")
        f.write(save)
        f.close()
        print("concept des documents preprocessé sauvegardé...")
    
    else:     
        print("Chargement du fichier json : preprocess_doc.json ...")
        with open(json_doc_preprocess) as json_file:
            all_document = json.load(json_file)
        print("Chargement du fichier json : preprocess_concept.json ...")
        with open(json_concept_preprocess) as json_file:
            all_concept = json.load(json_file)
        
        
    return all_document,all_concept

def count_concept(ac):
    all_concept = []
    for k in ac: 
        for w in ac[k]:
            all_concept.append(w)

    unique, counts = np.unique(all_concept, return_counts=True)
    dico_concept = dict(zip(unique, counts))
    return np.array(all_concept),dico_concept




ad,ac = pre_process_doc()
