import os
import sys
import time
import numpy as np
import cPickle as pickle

from simpleknn.bigfile import BigFile
from basic.common import printStatus, ROOT_PATH



INFO = os.path.basename(__file__)


class Synset2Vec:

    def __init__(self, corpus, modelName, wnid2words_file='data/wnid2words.pkl', rootpath=ROOT_PATH):
        printStatus(INFO + '.' + self.__class__.__name__, 'initializing ...')
        word2vec_dir = os.path.join(rootpath, corpus, 'word2vec', modelName)
        self.wnid2words = pickle.load(open(wnid2words_file, 'rb'))
        self.word2vec = BigFile(word2vec_dir)
    
  

    def get_feat_dim(self):
        return self.word2vec.ndims

        
    def explain(self, wnid):
        return self.wnid2words[wnid]

    def mapping(self, query_wnid):
        words = self.wnid2words[query_wnid].lower()
        words = [w.strip().replace(' ','_') for w in words.split(',')]
        words = [w.replace('-', '_') for w in words]
        for w in words:
            renamed, vectors = self.word2vec.read([w])
            if vectors:
                return vectors[0]
        return None


    def embedding(self, wnid):
        return self.mapping(wnid)



class PartialSynset2Vec (Synset2Vec):

    def mapping(self, wnid):
        words = self.wnid2words[wnid].lower().split(',')
        res = []
        for word in words:
            res += word.strip().replace('-', '_').split()

        word_vecs = []
        for w in res:
            renamed, vectors = self.word2vec.read([w])
            if vectors: 
                word_vecs.append(vectors[0])

        #print wnid, res, len(word_vecs)
        if len(word_vecs)>0:
            return np.array(word_vecs).mean(axis=0)
        else:
            return None
        
        


if __name__ == '__main__':
    rootpath = ROOT_PATH
    corpus = 'flickr4m'
    syn2vec = Synset2Vec(corpus, 'tagvec500', rootpath=rootpath)
    syn2vec2 = PartialSynset2Vec(corpus, 'tagvec500', rootpath=rootpath)
    queryset = str.split('n02084071 n04490091 n02114100 n03982060 n03219135 n05311054 n08615149 n02801525 n02330245')

    
    from simpleknn import simpleknn
    feat_dir = os.path.join(rootpath, corpus, 'word2vec', 'tagvec500')
    dim = syn2vec.word2vec.ndims
    nr_of_images = syn2vec.word2vec.nr_of_images 
    id_file = os.path.join(feat_dir, 'id.txt')
    searcher = simpleknn.load_model(os.path.join(feat_dir, "feature.bin"), dim, nr_of_images, id_file)

    
    for wnid in queryset:
        for s2v in [syn2vec, syn2vec2]:
            vec = s2v.embedding(wnid)
            print s2v, wnid, syn2vec.explain(wnid)
            for distance in ['l2']:
                searcher.set_distance(distance)
                visualNeighbors = searcher.search_knn(vec, max_hits=100)
                print wnid, distance, visualNeighbors[:10]
                print '-'*100



