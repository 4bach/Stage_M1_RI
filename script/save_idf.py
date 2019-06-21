from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import os
import pickle
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric,strip_multiple_whitespaces
table = str.maketrans('', '', '!"#$%\'()*+,-./:;<=>?@[\\]^_`{|}~')
CUSTOM_FILTERS = [strip_tags, strip_multiple_whitespaces]


def custom_tokenizer(s):
    return [w.translate(table) for w in preprocess_string(s, [strip_tags])]

def get_text(doc):
    return " ".join(preprocess_string(doc.text, CUSTOM_FILTERS))

folder_collection = "/local/karmim/Stage_M1_RI/data/collection"
folder_object = "/local/karmim/Stage_M1_RI/data/object_python"
corpus = []
for collection in ["FR94", "FT", "FBIS", "LATIMES"]:
    for root, dirs, files in os.walk(folder_collection+os.sep+collection, topdown=True):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                try:
                    filecontent = f.read()
                    soup = BeautifulSoup(filecontent, "html.parser")
                    docs = soup.find_all("doc")
                    for doc in docs:
                        corpus.append(get_text(doc))
                except:
                    continue

vectorizer = TfidfVectorizer(
                        use_idf=True,
                        smooth_idf=True, 
                        sublinear_tf=False,
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        ngram_range=(1,1), preprocessor=None,
                        stop_words=None, tokenizer=custom_tokenizer, vocabulary=None)

X = vectorizer.fit(corpus) #osef du transform
idf = vectorizer.idf_
idf = dict(zip(vectorizer.get_feature_names(), idf)) #notre dictionnaire terme -> idf

pickle.dump(idf, open(folder_object+os.sep+"idf_robust2004.pkl", "wb"))