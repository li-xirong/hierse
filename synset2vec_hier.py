import math
import os
import numpy as np
from basic.common import printStatus, ROOT_PATH
from synset2vec import Synset2Vec, PartialSynset2Vec

class WordnetHierarchy:

    def __init__(self, is_a_file='data/wordnet.is_a.txt'):
        parent_child = [x.strip().split() for x in open(is_a_file).readlines() if x.strip()]
        '''
        As the wordnet hierarcy is a net, some node may have more than one parent nodes.
        The following code simply makes each child has one parent.
        '''    
        self.child2parent = dict([(x[1],x[0]) for x in parent_child])

    
    def get_ancestors(self, wnid):
        ancestor_list = []
        parent = self.child2parent.get(wnid, None)
        if parent:
            return [parent] + self.get_ancestors(parent)
        else:
            assert ('n00001740' == wnid) # must be 'entity'
            return []


class HierSynset2Vec (Synset2Vec):

    def __init__(self, corpus, modelName, wnid2words_file='data/wnid2words.pkl', is_a_file='data/wordnet.is_a.txt', rootpath=ROOT_PATH):
        Synset2Vec.__init__(self, corpus, modelName, wnid2words_file, rootpath)
        self.hier = WordnetHierarchy(is_a_file)
        self.max_layer = 7

    def embedding(self, query_wnid):
        word2weight = {}
        ancestor_list = [query_wnid] + self.hier.get_ancestors(query_wnid)
        ancestor_list = ancestor_list[:self.max_layer] # At most go up to 7 ancestors

        weights = []
        word_vecs = []
    
        for layer,ance_id in enumerate(ancestor_list):
            vec = self.mapping(ance_id)
            if vec is not None:
                word_vecs.append(vec)
                weights.append( math.exp(-layer) )

        assert(len(weights)>0)        

        new_vec = (weights * np.matrix(word_vecs)).getA()[0] # np.array object
        Z = sum(weights)
        new_vec /= Z
        return new_vec


class HierPartialSynset2Vec (HierSynset2Vec, PartialSynset2Vec):

    def mapping(self, wnid):
        return PartialSynset2Vec.mapping(self, wnid)

    def embedding(self, query_wnid):
        return HierSynset2Vec.embedding(self, query_wnid)




NAME_TO_ENCODER = {'conse': Synset2Vec, 'conse2': PartialSynset2Vec, 'hierse': HierSynset2Vec, 'hierse2': HierPartialSynset2Vec}

def get_synset_encoder(name):
    return NAME_TO_ENCODER[name]



if __name__ == '__main__':
    rootpath = ROOT_PATH
    corpus = 'flickr4m'
    word2vec_model = 'tagvec500'

    syn2vec_list = []
    for name in str.split('conse conse2 hierse hierse2'):
        encoder_class = get_synset_encoder(name)
        syn2vec_list.append( encoder_class(corpus, word2vec_model, rootpath=rootpath) )

    queryset = str.split('n02084071 n04490091 n02114100 n03982060 n03219135 n05311054 n08615149 n02801525 n02330245')

    wnhier = WordnetHierarchy()
    for wnid in queryset:
        ancestor_list = wnhier.get_ancestors(wnid)
        print wnid, syn2vec_list[0].explain(wnid)
        print [syn2vec_list[0].explain(x) for x in ancestor_list]
        print ''


    from simpleknn import simpleknn
    feat_dir = os.path.join(rootpath, corpus, 'word2vec', word2vec_model)
    dim = syn2vec_list[0].word2vec.ndims
    nr_of_images = syn2vec_list[0].word2vec.nr_of_images 
    id_file = os.path.join(feat_dir, 'id.txt')
    searcher = simpleknn.load_model(os.path.join(feat_dir, "feature.bin"), dim, nr_of_images, id_file)

    
    for wnid in queryset:
        for s2v in syn2vec_list:
            vec = s2v.embedding(wnid)
            print s2v, wnid, s2v.explain(wnid)
            for distance in ['l2']:
                searcher.set_distance(distance)
                visualNeighbors = searcher.search_knn(vec, max_hits=100)
                print wnid, distance, visualNeighbors[:10]
                print '-'*100


