import torch 
import numpy as np 
import pickle 

"""
    auteur -> Isma le plus beau rebeu 100% Chaoui .
"""


class Reranker:
    def __init__(self, bm25_dict,histo_type='LCH',folder_interaction="/local/karmim/Stage_M1_RI/data/object_python/interaction/all_robust/"):
        self.bm25_dict = bm25_dict #le dictionnaire query -> 2000 docs relevants pour bm25
        self.folder_interaction=folder_interaction
        self.histo_type=histo_type
    def set_model(self, model):
        self.model = model
    
    def rerank(self, queries=None,idf_file="/local/karmim/Stage_M1_RI/data/object_python/idf_robust2004.pkl"):
        
        #queries: une liste d'ID de requete pour lesquels on veut réordonner les résultats
        if queries == None:
            queries_to_rerank = list(self.bm25_dict.keys())
        else:
            queries_to_rerank = queries
        
        #pour chaque requete, on va charger réordonner ses résultats avec le modèle
        query_idf = pickle.load(open(idf_file, "rb"))
        reranked_dict = {}
        for id_requete in queries_to_rerank:
            if id_requete != "634":
                with torch.no_grad():
                    #contient une matrice (2000, query_max_len, hist_size)
                    saintjeanlapuenta = np.load(self.folder_interaction+id_requete+self.histo_type+"_interractions.npy")
                    a = np.tile(np.array([query_idf[id_requete]]), (saintjeanlapuenta.shape[0],1))

                    model_scores = self.model(saintjeanlapuenta, a)
                    lol = np.argsort(model_scores)[::-1] #tri décroissant

                    # reranked: liste de tuples (document_id, score)
                    reranked_dict[id_requete] = [(self.bm25_dict[id_requete][i][0], model_scores[i]) for i in lol]

        return reranked_dict
    
    def get_results(self, id_requete, rank_list):
        results = []
        for i, (doc_id, score) in enumerate(rank_list[:1000]):
            results.append(" ".join([id_requete, "Q0", doc_id, str(i + 1), str(score), "EARIA"]))
        return results
            
    
    def save_results(self, rank_dict, res_file):
        """
        sauver sur un fichier au format attendu par TREC
        un dictionnaire query_id -> list (doc_id, score)
        """
        results = [f"{id_requete} Q0 EMPTY 1001 -100000 EARIA" for id_requete in rank_dict]
        for id_requete in rank_dict:
            results.extend(self.get_results(id_requete, rank_dict[id_requete]))
        
        with open(res_file, "w") as tiacompris:
            tiacompris.write("\n".join(results))