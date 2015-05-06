import os
import sys
import random
import numpy as np
import unittest

from basic.common import printStatus, ROOT_PATH
from basic.util import readImageSet

from simpleknn.bigfile import BigFile
from tagger import ZeroshotTagger
from im2vec import Image2Vec
from eval import HitScorer


INFO = os.path.basename(__file__)
rootpath = ROOT_PATH


class TestSuite (unittest.TestCase):

    def test_rootpath(self):
        self.assertTrue(os.path.exists(rootpath))


    def test_datafiles(self):
        self.assertTrue(os.path.exists('data/ilsvrc12/synsets.txt'))
        self.assertTrue(os.path.exists('data/ilsvrc12/synsets2hop.txt'))
        self.assertTrue(os.path.exists('data/wnid2words.pkl'))
        self.assertTrue(os.path.exists('data/wordnet.is_a.txt'))
        self.assertTrue(os.path.exists('data/imagenet2hop-random2k.tar.gz'))


    def test_testdata(self):
        os.system('tar xzf data/imagenet2hop-random2k.tar.gz -C %s' % rootpath)
        imsetfile = os.path.join(rootpath, 'imagenet2hop-random2k', 'ImageSets', 'imagenet2hop-random2k.txt')
        self.assertTrue(os.path.exists(imsetfile), msg='imagenet2hop-random2k is not ready')


    def test_word2vec(self):
        shape_file = os.path.join(rootpath, 'flickr4m', 'word2vec', 'tagvec500', 'shape.txt')
        self.assertTrue(os.path.exists(shape_file), msg='word2vec is not ready')


    def test_tagging(self):
        corpus = 'flickr4m'
        word2vec_model = 'tagvec500'
        testCollection = 'imagenet2hop-random2k'
        imset = readImageSet(testCollection, testCollection, rootpath)
        feature = 'dascaffeprob'

        feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', feature))
        blocksize = 1000
        scorers = [HitScorer(n) for n in [1, 2, 5, 10]]

        overwrite = 1

        for embedding_model in str.split('conse conse2 hierse hierse2'):
            embedding_name = '%s,%s,%s' % (corpus, word2vec_model, embedding_model)

            for synset_name in str.split('imagenet1k imagenet1k2hop'):
                if 'imagenet1k' == synset_name:
                    label_file = 'data/ilsvrc12/synsets.txt'
                else:
                    label_file = 'data/ilsvrc12/synsets2hop.txt'

                params = '%s %s --embedding %s --word2vec %s --corpus %s --overwrite %d' % (label_file, synset_name, embedding_model, word2vec_model, corpus, overwrite)
                os.system('python build_synset_vec.py %s' % params)
                shape_file = os.path.join(rootpath, 'synset2vec', synset_name, embedding_name, 'shape.txt')
                self.assertTrue(os.path.exists(shape_file), msg="%s is not ready" % synset_name)

    
            synset_name = 'imagenet1k'
            label_file = 'data/ilsvrc12/synsets.txt'
            label2vec_dir = os.path.join(rootpath, 'synset2vec', synset_name, embedding_name)
            i2v = Image2Vec(label_file, label2vec_dir)

            tagger = ZeroshotTagger(embedding_name = embedding_name)
            printStatus(INFO, 'tagging %d images' % len(imset))

            start = 0

    
            overall_perf = [0.0] * len(scorers)
            nr_of_images = 0

            while start < len(imset):
                end = min(len(imset), start + blocksize)
                renamed, vectors = feat_file.read(imset[start:end])

                for _id,_vec in zip(renamed, vectors):
                    truth = set([_id.split('_')[0]])
                    im_vec = i2v.embedding(_vec)
                    pred = tagger.predict(im_vec)
                    sorted_labels = [int(x[0] in truth) for x in pred]
                    perf = [scorer.score(sorted_labels) for scorer in scorers]
                    overall_perf = [overall_perf[i] + perf[i] for i in range(len(scorers))]
                    nr_of_images += 1

                start = end
    
            res = [x/nr_of_images for x in overall_perf]
            print '_'*100
            print embedding_name
            print ' '.join([x.name() for x in scorers])
            print ' '.join(['%.3f' % x for x in res])
            print '_'*100



suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)



    












