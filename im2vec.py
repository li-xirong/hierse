# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import numpy as np
import logging

from constant import *
import utility
from simpleknn.bigfile import BigFile


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Image2Vec:

    def __init__(self, Y0=DEFAULT_Y0, label_vec_name=DEFAULT_LABEL_VEC_NAME, rootpath=ROOT_PATH):
        label_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/synsets_%s.txt' % Y0)
        label2vec_dir = os.path.join(rootpath, 'synset2vec', Y0, label_vec_name)

        self.labels = map(str.strip, open(label_file).readlines())
        self.nr_of_labels = len(self.labels)

        feat_file = BigFile(label2vec_dir)
        renamed, vectors = feat_file.read(self.labels)
        name2index = dict(zip(renamed, range(len(renamed))))
        self.label_vectors = [None] * self.nr_of_labels
        self.feat_dim = feat_file.ndims

        for i in xrange(self.nr_of_labels):
            idx = name2index.get(self.labels[i], -1)
            self.label_vectors[i] = np.array(vectors[idx]) if idx >= 0 else None

        nr_of_inactive_labels = len([x for x in self.label_vectors if x is None])    
        logger.info('#active_labels=%d, embedding_size=%d', self.nr_of_labels - nr_of_inactive_labels, self.feat_dim)


    def embedding(self, prob_vec, k=DEFAULT_K):
        assert(len(prob_vec) == self.nr_of_labels), 'len(prob_vec)=%d, nr_of_labels=%d' % (len(prob_vec), self.nr_of_labels)
        top_hits = np.argsort(prob_vec)[::-1][:k]
        new_vec = np.array([0.] * self.feat_dim)

        Z = 0.
        for idx in top_hits:
            vec = self.label_vectors[idx]
            if vec is not None:
                new_vec += prob_vec[idx] * vec
                Z += prob_vec[idx]
        if Z > 1e-10:
            new_vec /= Z
        return new_vec




def process(options, image_collection, pY0):
    rootpath = options.rootpath
    overwrite = options.overwrite
    k = options.k
    batch_size = options.batch_size
    subset = options.subset if options.subset else image_collection
    Y0 = options.Y0
    label_vec_name = options.label_vec_name
    new_feature = '%s,%s,%s' % (Y0, label_vec_name, pY0)

    resfile = os.path.join(rootpath, image_collection, 'FeatureData', new_feature, 'id.feature.txt')
    if os.path.exists(resfile) and not overwrite:
        logger.info('%s exists. quit', resfile)
        return 0

    imsetfile = os.path.join(rootpath, image_collection, 'ImageSets', '%s.txt' % subset)
    imset = map(str.strip, open(imsetfile).readlines())
    logger.info('%d images to do', len(imset))

    feat_file = BigFile(os.path.join(rootpath, image_collection, 'FeatureData', pY0))

    im2vec = Image2Vec(Y0, label_vec_name, rootpath)

    utility.makedirsforfile(resfile)
    fw = open(resfile, 'w')

    read_time = 0
    run_time = 0
    start = 0
    done = 0

    while start < len(imset):
        end = min(len(imset), start + batch_size)
        logger.info('processing images from %d to %d', start, end-1)

        s_time = time.time()
        renamed, test_X = feat_file.read(imset[start:end])
        read_time += time.time() - s_time
        
        s_time = time.time()
        output = [None] * len(renamed)
        for i in xrange(len(renamed)):
            vec = im2vec.embedding(test_X[i], k)
            output[i] = '%s %s\n' % (renamed[i], " ".join(map(str, vec)))
        run_time += time.time() - s_time
        start = end
        fw.write(''.join(output))
        done += len(output)

    # done    
    fw.close()
    logger.info("%d done. read time %g seconds, run_time %g seconds", done, read_time, run_time)
    return done


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] image_collection pY0""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--subset", default="", type="string", help="only do this subset")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="top-k labels used for semantic embedding (default: %d)" % DEFAULT_K)
    parser.add_option("--batch_size", default=DEFAULT_BATCH_SIZE, type="int", help="nr of feature vectors loaded into memory (default: %d)" % DEFAULT_BATCH_SIZE)
    parser.add_option("--Y0", default=DEFAULT_Y0, type="string", help="name ofthe Y0 label set (default: %s)" % DEFAULT_Y0)
    parser.add_option("--label_vec_name", default=DEFAULT_LABEL_VEC_NAME, type="string", help="precomputed w2v vectors of the Y0 label set (default: %s)" % DEFAULT_LABEL_VEC_NAME)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
