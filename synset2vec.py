from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import cPickle as pickle
import logging

from simpleknn.bigfile import BigFile
from constant import *

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

class Synset2Vec:

    def __init__(self, corpus=DEFAULT_W2V_CORPUS, w2v_name=DEFAULT_W2V, wnid2words_file=DEFAULT_WNID2WORDS_FILE, rootpath=ROOT_PATH):
        word2vec_dir = os.path.join(rootpath, corpus, 'word2vec', w2v_name)
        self.word2vec = BigFile(word2vec_dir)
        self.wnid2words = pickle.load(open(wnid2words_file, 'rb'))
        logger.info('w2v(%s): %d words, %d dims', corpus, self.word2vec.shape()[0], self.get_feat_dim())

    def get_feat_dim(self):
        return self.word2vec.ndims

        
    def explain(self, wnid):
        return self.wnid2words[wnid]

    def _mapping(self, query_wnid):
        words = self.wnid2words[query_wnid].lower()
        words = [w.strip().replace(' ','_') for w in words.split(',')]
        words = [w.replace('-', '_') for w in words]
        for w in words:
            vec = self.word2vec.read_one(w)
            if vec:
                return vec
        return None


    def embedding(self, wnid):
        return self._mapping(wnid)



class PartialSynset2Vec (Synset2Vec):

    def _mapping(self, wnid):
        words = self.wnid2words[wnid].lower().split(',')
        res = []
        for word in words:
            res += word.strip().replace('-', '_').split()

        word_vecs = []
        for w in res:
            vec = self.word2vec.read_one(w)
            if vec:
                word_vecs.append(vec)

        if len(word_vecs)>0:
            return np.array(word_vecs).mean(axis=0)
        else:
            return None
        
        


if __name__ == '__main__':
    rootpath = ROOT_PATH
    syn2vec = Synset2Vec()
    syn2vec2 = PartialSynset2Vec()
    queryset = str.split('n02084071 n04490091 n02114100 n03982060 n03219135 n05311054 n08615149 n02801525 n02330245')

    
    from simpleknn import simpleknn
    feat_dir = os.path.join(rootpath, 'flickr4m', 'word2vec', 'tagvec500')
    searcher = simpleknn.load_model(feat_dir)

    
    for wnid in queryset:
        for s2v in [syn2vec, syn2vec2]:
            vec = s2v.embedding(wnid)
            print (s2v, wnid, syn2vec.explain(wnid))
            for distance in ['cosine']:
                searcher.set_distance(distance)
                visualNeighbors = searcher.search_knn(vec, max_hits=100)
                print (wnid, distance, visualNeighbors[:10])
                print ('-'*100)



