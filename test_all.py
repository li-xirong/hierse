from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import unittest

from constant import *
from simpleknn.bigfile import BigFile
from tagger import ZeroshotTagger
from im2vec import Image2Vec
from evaluate import HitScorer


rootpath = ROOT_PATH
TEST_COLLECTION = 'imagenet2hop-random2k'


class TestSuite (unittest.TestCase):

    def test_rootpath(self):
        self.assertTrue(os.path.exists(rootpath))


    def test_datafiles(self):
        self.assertTrue(os.path.exists('data/synsets_%s.txt' % DEFAULT_Y0))
        self.assertTrue(os.path.exists('data/synsets_%s.txt' % DEFAULT_Y1))
        self.assertTrue(os.path.exists('data/wnid2words.pkl'))
        self.assertTrue(os.path.exists('data/wordnet.is_a.txt'))
        self.assertTrue(os.path.exists('data/%s.tar.gz' % TEST_COLLECTION))


    def test_testdata(self):
        os.system('tar xzf data/%s.tar.gz -C %s' % (TEST_COLLECTION, rootpath))
        imsetfile = os.path.join(rootpath, TEST_COLLECTION, 'ImageSets', '%s.txt'%TEST_COLLECTION)
        self.assertTrue(os.path.exists(imsetfile), msg='%s is not ready' % TEST_COLLECTION)


    def test_word2vec(self):
        shape_file = os.path.join(rootpath, DEFAULT_W2V_CORPUS, 'word2vec', DEFAULT_W2V, 'shape.txt')
        self.assertTrue(os.path.exists(shape_file), msg='pretrained w2v model is not ready')


    def test_zeroshot_tagging(self):
        corpus = DEFAULT_W2V_CORPUS
        word2vec_model = DEFAULT_W2V
        image_collection = TEST_COLLECTION
        
        imsetfile = os.path.join(rootpath, image_collection, 'ImageSets', '%s.txt'%image_collection)
        imset = map(str.strip, open(imsetfile).readlines())

        feat_file = BigFile(os.path.join(rootpath, image_collection, 'FeatureData', DEFAULT_pY0))
        batch_size = 1000
        scorers = [HitScorer(n) for n in [1, 2, 5, 10]]

        overwrite = 1
        Y0 = DEFAULT_Y0
        Y1 = DEFAULT_Y1
    

        for embedding in str.split('conse conse2 hierse hierse2'):
            label_vec_name = '%s,%s,%s' % (corpus, word2vec_model, embedding)

            for synset_name in [Y0, Y1]:
                os.system('python build_label_vec.py %s --embedding %s --overwrite 1' % (synset_name, embedding))
                shape_file = os.path.join(rootpath, 'synset2vec', synset_name, label_vec_name, 'shape.txt')
                self.assertTrue(os.path.exists(shape_file), msg="%s is not ready" % synset_name)

            i2v = Image2Vec(Y0=Y0, label_vec_name=label_vec_name)

            tagger = ZeroshotTagger(Y1=Y1, label_vec_name=label_vec_name)
            print ('tagging %d images' % len(imset))

            start = 0

    
            overall_perf = [0.0] * len(scorers)
            nr_of_images = 0

            while start < len(imset):
                end = min(len(imset), start + batch_size)
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
            print ('_'*100)
            print (embedding)
            print (' '.join([x.name() for x in scorers]))
            print (' '.join(['%.3f' % x for x in res]))
            print ('_'*100)



suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)



    












