import sys
sys.path.insert(0, '/local/karmim/Stage_M1_RI/src')
import load_data
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

data = load_data.Dataset()
relevance = data.load_relevance()
docs = data.load_all_docs()
que_emb = data.embedding_query()


cosine_similarity(data.model_wv['car'].reshape(1,-1),data.model_wv['truck'].reshape(1,-1)).item()
hist = np.array([cosine_similarity(que_emb['301'][0].reshape(1,-1),data.model_wv[i].reshape(1,-1)).item() for i in docs['LA070389-0001']]) 


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

def calcul_interaction(self,query,document,bins_=4,normalize=False):
    """
        Fonction qui calcul un histogramme d'une interaction cosinus similarité entre une query et un doc. 
        Entrée -> Embedding d'une query et d'un document.
                    bins : Nombre d'intervalles dans l'histogramme.
                    normalize : Bool pour normaliser l'histogramme. 
        Sortie -> Renvoie une matrice d'histogramme.
    """
    X = []

    # Calcul de similarité entre doc et query 
    for q in query:
        histo = []
        for d in document:
            histo.append(cosine_similarity(q, d)[0][0])
        histo, _ = np.histogram(histo, bins= bins_)
        if normalize:
            histo = histo / histo.sum()
        X.append(histo)
    return np.array(X)

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
