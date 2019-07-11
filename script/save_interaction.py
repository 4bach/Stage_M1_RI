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
if not sys.warnoptions:
    warnings.simplefilter("ignore")


data = load_data.Dataset()
relevance = data.load_relevance()
docs = data.load_all_docs()
query = data.load_all_query()
que_emb = data.embedding_query()
max_length = data.max_length_query
cosine_similarity(data.model_wv['car'].reshape(1,-1),data.model_wv['truck'].reshape(1,-1)).item()
hist = np.array([cosine_similarity(que_emb['301'][0].reshape(1,-1),data.model_wv[i].reshape(1,-1)).item() for i in docs['LA070389-0001']]) 



def histo( query,doc_id,data,intervals=30,max_length=5,histo_type='CH'):
    """
        for a query embedded, a doc_id and the data object return the interaction histogram matrix. 
            intervals -> number of bins in a histogram
            histo_type -> The three types of histograms in DRMM original architecure : CH, NH, LCH.

    """

    #embedded docs 
    doc_emb = []
    cpt = 0
    for i,w in enumerate(data.docs[doc_id]): 
        #print(w) 
        #print(type(w)) 
        if  w in data.model_wv: 
            doc_emb.append( data.model_wv[w] )                 
        elif w.upper() in data.model_wv:
            doc_emb.append( data.model_wv[w.upper()] )
        elif w.lower() in data.model_wv:
            doc_emb.append(data.model_wv[w.lower()])
        else : 
            cpt+=1

    doc_emb = np.array(doc_emb)
    if histo_type=='LCH':
        cos = cosine_similarity(query,doc_emb)
        mat_hist = np.array([np.log([np.histogram(cos[j],bins=intervals)[0] if (j < len(query))  else np.zeros((intervals,)) for j in range(max_length)])])
        mat_hist[mat_hist < 0] = 0
    else:
        cos = cosine_similarity(query,doc_emb)
        mat_hist = np.array([np.histogram(cos[j],bins=intervals)[0] if j < len(query) else np.zeros((intervals,)) for j in range(max_length)])

        if histo_type == 'NH':
            mat_hist = np.array([i/i.sum() if i.sum()!= 0 else np.zeros(np.shape(i)) for i in mat_hist])
        
    return mat_hist 


ch = histo(que_emb['301'],'LA070389-0001',data)
lch = histo(que_emb['301'],'LA070389-0001',data,histo_type='LCH') # Work well for the DRMM architecture. 
nh = histo(que_emb['301'],'LA070389-0001',data,histo_type='NH') # Dont work really good in the original paper.

def calcul_all_interaction_forNN(data,intervals = 30,histo_type='CH',train_size=0.8,folder_interaxion_np ="/local/karmim/Stage_M1_RI/data/object_python/interaction"):
    """
        This function compute all the interaction needed to train our Neural Network.  
            Input -> 
                data : object from the file load_data.py 
                intervals : number of bins in the histogram 
                histo_type : The type of the histogram : CH, NH or LCH.
                train_size : % of query that are use for the train   
            Output -> List of interaction matrix, with pos and neg examples. 
    """
    
    relevance = data.load_relevance()
    #docs = data.load_all_docs()
    #query = data.load_all_query()
    que_emb = data.embedding_query()
    max_length = data.max_length_query
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
        exists = os.path.isfile(folder_interaxion_np+id_q+"_interractions.npy")
        if not exists : 
            all_interaxion_4_query = []
            pos = np.array([histo(que_emb[id_q],doc_pos,data,intervals=intervals,max_length=max_length,histo_type=histo_type) for doc_pos in relevance[id_q]['relevant']])
            all_interaxion_4_query.append(pos)
            neg = np.array([histo(que_emb[id_q],doc_neg,data,intervals=intervals,max_length=max_length,histo_type=histo_type) for doc_neg in relevance[id_q]['irrelevant']])
            all_interaxion_4_query.append(neg)
            all_interaxion_4_query = np.array(all_interaxion_4_query)
            train_X[id_q] = all_interaxion_4_query
            np.save(folder_interaxion_np+id_q+"_interractions.npy", all_interaxion_4_query)
            print(folder_interaxion_np+id_q+"_interractions.npy succesfully saved.")
        else:
            all_interaxion_4_query = np.load(folder_interaxion_np+id_q+"_interractions.npy")
            train_X[id_q] = all_interaxion_4_query
            print(folder_interaxion_np+id_q+"_interractions.npy succesfully loaded.")

    for id_q in test_q:
        exists = os.path.isfile(folder_interaxion_np+id_q+"_interractions.npy")
        if not exists : 
            all_interaxion_4_query = []
            pos = np.array([histo(que_emb[id_q],doc_pos,data,intervals=intervals,max_length=max_length,histo_type=histo_type) for doc_pos in relevance[id_q]['relevant']])
            all_interaxion_4_query.append(pos)
            neg = np.array([histo(que_emb[id_q],doc_neg,data,intervals=intervals,max_length=max_length,histo_type=histo_type) for doc_neg in relevance[id_q]['irrelevant']])
            all_interaxion_4_query.append(neg)
            all_interaxion_4_query = np.array(all_interaxion_4_query)
            test_X[id_q] = all_interaxion_4_query
            np.save(folder_interaxion_np+id_q+"_interractions.npy", all_interaxion_4_query)
            print(folder_interaxion_np+id_q+"_interractions.npy succesfully saved.")
        else:
            all_interaxion_4_query = np.load(folder_interaxion_np+id_q+"_interractions.npy")
            test_X[id_q] = all_interaxion_4_query
            print(folder_interaxion_np+id_q+"_interractions.npy succesfully loaded.")
    return train_X,test_X


#calcul_all_interaction_forNN(data,intervals = 30,histo_type='CH',train_size=0.8,folder_interaxion_np ="/local/karmim/Stage_M1_RI/data/object_python/interaction")

def prepare_data_forNN(self, test_size=0.2):
        """
        """
        pass
    # #spliter les requêtes en train/test
    # lol = [q for q in self.d_query.keys() if q in self.paires]
    # random.shuffle(lol)
    # test_keys = lol[:int(test_size * len(lol))]
    # train_keys = lol[int(test_size * len(lol)):]
    
    # #pour chaque requête on va générer autant de paires relevant que irrelevant
    # #pour nos besoins on va alterner paires positives et paires négatives
    # train_hist = [] # les histogrammes d'interraction
    # test_hist = []
    # train_idf = [] #les vecteurs d'idf
    # test_idf = []
    
    # for id_requete in train_keys:
    #     #recuperer les mots dont on connait les embeddings dans la query
    #     q = self.vectoriser.transform([' '.join(self.d_query[id_requete])])
    #     idf_vec = self.get_idf_vec(self.d_query[id_requete])
    #     for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
    #         #lire le doc, la requete et creer l'histogramme d'interraction
            
    #         d = self.vectoriser.transform([' '.join(self.docs[pos])])
    #         train_hist.append(self.hist(q, d)) #append le doc positif
    #         train_idf.append(idf_vec) #append le vecteur idf de la requête
            
            
    #         d = self.vectoriser.transform([' '.join(self.docs[neg])])
    #         train_hist.append(self.hist(q, d)) #append le doc négatif
    #         train_idf.append(idf_vec) #append le vecteur idf de la requête
    # train_labels = np.zeros(len(train_hist))
    # train_labels[::2] = 1 # label de pertinence 
    
    
    # for id_requete in test_keys:
    #     #recuperer les mots dont on connait les embeddings dans la query
    #     q = self.vectoriser.transform([' '.join(self.d_query[id_requete])])
    #     idf_vec = self.get_idf_vec(self.d_query[id_requete])
    #     for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
    #         #lire le doc, la requete et creer l'histogramme d'interraction
            
    #         d = self.vectoriser.transform([' '.join(self.docs[pos])])

    #         test_hist.append(self.hist(q, d)) #append le doc positif
    #         test_idf.append(idf_vec) #append le vecteur idf de la requête
            
    #         d = self.vectoriser.transform([' '.join(self.docs[neg])])


    #         test_hist.append(self.hist(q, d)) #append le doc négatif
    #         test_idf.append(idf_vec) #append le vecteur idf de la requête
    # test_labels = np.zeros(len(train_hist))
    # test_labels[::2] = 1
    
    # return (train_hist, train_idf, train_labels), (test_hist, test_idf, test_labels)
    
    # #éventuellement sauvegarder tout ça sur le disque comme ça c fait une bonne fois pour toutes...
