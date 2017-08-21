from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import logging

from constant import *
import utility
from synset2vec_hier import get_synset_encoder

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def process(options, label_set):
    overwrite = options.overwrite
    rootpath = options.rootpath
    w2v_corpus = options.w2v_corpus
    w2v = options.w2v
    embedding = options.embedding
    
    resdir = os.path.join(rootpath, 'synset2vec', label_set, '%s,%s,%s' % (w2v_corpus, w2v, embedding))
    resfile = os.path.join(resdir, 'feature.bin')
    if os.path.exists(resfile) and not overwrite:
        logger.info('%s exists. quit', resfile)
        return 0

    synset_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'synsets_%s.txt' % label_set)
    synsets = map(str.strip, open(synset_file).readlines())
    s2v = get_synset_encoder(embedding)(w2v_corpus, w2v, rootpath=rootpath)
  
    utility.makedirsforfile(resfile)

    good = []
    with open(resfile, 'wb') as fw:
        for i,wnid in enumerate(synsets):
            #if i % 1e3 == 0:
            #    printStatus(INFO, '%d done' % i)
            vec = s2v.embedding(wnid)

            if vec is not None:
                vec = np.array(vec, dtype=np.float32)
                vec.tofile(fw)
                good.append(wnid)

        fw.close()
        logger.info('%d done, %d okay' % ((i+1), len(good)))
 

    with open(os.path.join(resdir, 'id.txt'), 'w') as fw:
        fw.write(' '.join(good))
        fw.close()

    with open(os.path.join(resdir, 'shape.txt'), 'w') as fw:
        fw.write('%d %d' % (len(good), s2v.get_feat_dim()))
        fw.close() 



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] label_set""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--w2v_corpus", default=DEFAULT_W2V_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_W2V_CORPUS)
    parser.add_option("--w2v", default=DEFAULT_W2V, type="string", help="word2vec model (default: %s)" % DEFAULT_W2V)
    parser.add_option("--embedding", default=DEFAULT_EMBEDDING, type="string", help="embedding model (default: %s)" % DEFAULT_EMBEDDING)
    
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())
