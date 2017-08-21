from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import logging

from constant import *
from simpleknn.bigfile import BigFile
import utility
from tagger import ZeroshotTagger
from im2vec import Image2Vec

DEFAULT_R = 20

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def process(options, testCollection):
    overwrite = options.overwrite
    rootpath = options.rootpath
    
    Y0 = options.Y0
    Y1 = options.Y1
    pY0 = options.pY0
    r = options.r

    batch_size = 2000

    label_vec_name = '%s,%s,%s' % (options.w2v_corpus, options.w2v, options.embedding)
    for synset_name in [Y0, Y1]:
        assert(os.path.exists(os.path.join(rootpath, 'synset2vec', synset_name, label_vec_name)))

    resfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, pY0, label_vec_name, 'id.tagvotes.txt')

    if os.path.exists(resfile) and not overwrite:
        logger.info('%s exists. quit', resfile)
        return 0

    i2v = Image2Vec(Y0=Y0, label_vec_name=label_vec_name)
    tagger = ZeroshotTagger(Y1=Y1, label_vec_name=label_vec_name, rootpath=rootpath)

    imset = utility.readImageSet(testCollection, testCollection, rootpath)
    feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', pY0)
    feat_file = BigFile(feat_dir)
    

    logger.info('tagging %d images', len(imset))
    utility.makedirsforfile(resfile)
    logger.info('save results to %s', resfile)

    fw = open(resfile, 'w')

    start = 0
    while start < len(imset):
        end = min(len(imset), start + batch_size)
        logger.info('processing images from %d to %d', start, end)
        todo = imset[start:end]
        if not todo:
            break

        renamed, vectors = feat_file.read(todo)
        output = []
        for _id,_vec in zip(renamed, vectors):
            im_vec = i2v.embedding(_vec)
            pred = tagger.predict(im_vec, topk=options.r)
            output.append('%s %s\n' % (_id, ' '.join(['%s %s'%(x[0],x[1]) for x in pred])))
        start = end
        fw.write(''.join(output))

    fw.close()



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--w2v_corpus", default=DEFAULT_W2V_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_W2V_CORPUS)
    parser.add_option("--w2v", default=DEFAULT_W2V, type="string", help="word2vec model (default: %s)" % DEFAULT_W2V)
    parser.add_option("--embedding", default=DEFAULT_EMBEDDING, type="string", help="embedding model (default: %s)" % DEFAULT_EMBEDDING)
    parser.add_option("--Y0", default=DEFAULT_Y0, type="string", help="training labels (default: %s)" % DEFAULT_Y0)
    parser.add_option("--Y1", default=DEFAULT_Y1, type="string", help="test labels (default: %s)" % DEFAULT_Y1)
    parser.add_option("--pY0", default=DEFAULT_pY0, type="string", help="probabilistic prediction of Y0 (default: %s)" % DEFAULT_pY0)
    parser.add_option("--r", default=DEFAULT_R, type="int", help="how many tags to predict (default: %d)" % DEFAULT_R)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    assert(DEFAULT_Y0 == options.Y0)
    assert(DEFAULT_Y1 == options.Y1)
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())
