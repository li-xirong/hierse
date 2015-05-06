import os
import sys
import numpy as np

from basic.common import printStatus, ROOT_PATH
from simpleknn.bigfile import BigFile


INFO = os.path.basename(__file__)


def cosine_similarity(vecx, vecy):
    norm = np.sqrt(np.dot(vecx, vecx))* np.sqrt(np.dot(vecy, vecy))
    return np.dot(vecx, vecy) / (norm + 1e-10)


class ZeroshotTagger:

    def __init__(self, synset_name='imagenet1k2hop', embedding_name='flickr4m,tagvec500,hierse2', rootpath=ROOT_PATH):
        feat_dir = os.path.join(rootpath, 'synset2vec', synset_name, embedding_name)
        feat_file = BigFile(feat_dir)
        self.labels = feat_file.names
        self.nr_of_labels = len(self.labels)
        self.feat_dim = feat_file.ndims

        renamed, vectors = feat_file.read(self.labels)
        name2index = dict(zip(renamed, range(len(renamed))))
        self.label_vectors = [None] * self.nr_of_labels
        
        for i in xrange(self.nr_of_labels):
            idx = name2index.get(self.labels[i], -1)
            self.label_vectors[i] = np.array(vectors[idx]) if idx >= 0 else None

        nr_of_inactive_labels = len([x for x in self.label_vectors if x is None])    
        printStatus(INFO + '.' + self.__class__.__name__, '#active_labels=%d, embedding_size=%d' % (self.nr_of_labels - nr_of_inactive_labels, self.feat_dim))


    def _compute(self, img_vec):
        scores = [0] * self.nr_of_labels
        for i in xrange(self.nr_of_labels):
            scores[i] = cosine_similarity(img_vec, self.label_vectors[i]) if self.label_vectors[i] is not None else -1
        return scores


    def predict(self, img_vec, topk=20):
        scores = self._compute(img_vec)
        sorted_idx = np.argsort(scores)[::-1][:topk]
        return [(self.labels[i], scores[i]) for i in sorted_idx]



if __name__ == '__main__':
    rootpath = ROOT_PATH

    embedding_model = 'hierse2'
    embedding_name = 'flickr4m,tagvec500,%s' % embedding_model
    tagger = ZeroshotTagger(embedding_name = embedding_name)
    label_file = 'data/ilsvrc12/synsets.txt'
    label2vec_dir = os.path.join(rootpath, 'synset2vec', 'imagenet1k', embedding_name)
    from im2vec import Image2Vec
    i2v = Image2Vec(label_file, label2vec_dir)

    from basic.util import readImageSet
    testCollection = 'imagenet2hop'
    imset = readImageSet(testCollection, 'random100k', rootpath)
    feature = 'dascaffeprob'
    feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', feature))

    blocksize = 1000
    start = 0

    from eval import HitScorer

    scorers = [HitScorer(n) for n in [1, 2, 5, 10]]
    overall_perf = [0.0] * len(scorers)
    nr_of_images = 0

    while start < len(imset):
        end = min(len(imset), start + blocksize)
        renamed, vectors = feat_file.read(imset[start:end])

        for _id,_vec in zip(renamed, vectors):
            truth = set([_id.split('_')[0]])
            im_vec = i2v.embedding(_vec)
            res = tagger.predict(im_vec)
            sorted_labels = [int(x[0] in truth) for x in res]
            perf = [scorer.score(sorted_labels) for scorer in scorers]
            overall_perf = [overall_perf[i] + perf[i] for i in range(len(scorers))]
            nr_of_images += 1
        break

    res = [x/nr_of_images for x in overall_perf]
    print ' '.join([x.name() for x in scorers])
    print ' '.join(['%.3f' % x for x in res])












