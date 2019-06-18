from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.insert(0, '/local/karmim/Stage_M1_RI/src')
import load_data



data = load_data.Dataset(30,normalize=False)
voc = data.get_vocab()
vocabulary = [w for w in voc]
vectorizer = CountVectorizer(analyzer='word', vocabulary=vocabulary, binary=True, lowercase=False)
tf_idf_path = '/local/karmim/Stage_M1_RI/data/object_python/idf_robust2004.pkl'
data.set_params(vectorizer,tf_idf_path)

data.load_all_query()
data.load_relevance()
data.load_docs_per_folder()

data.get_query('302')
data.get_doc('FBIS4-11528')
data.get_relevance('302')


