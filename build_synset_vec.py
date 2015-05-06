import sys
import os
import numpy as np
from synset2vec_hier import get_synset_encoder
from basic.common import checkToSkip, makedirsforfile, niceNumber, ROOT_PATH, printStatus

DEFAULT_CORPUS = 'flickr4m'
DEFAULT_WORD2VEC = 'tagvec500'
DEFAULT_EMBEDDING = 'hierse2'


INFO = os.path.basename(__file__)


def process(options, synset_file, synset_name):
    overwrite = options.overwrite
    rootpath = options.rootpath
    corpus = options.corpus
    word2vec_model = options.word2vec
    embedding = options.embedding

    resdir = os.path.join(rootpath, 'synset2vec', synset_name, '%s,%s,%s' % (corpus, word2vec_model, embedding))
    resfile = os.path.join(resdir, 'feature.bin')
    if checkToSkip(resfile, overwrite):
        return 0

    synsets = map(str.strip, open(synset_file).readlines())
    s2v = get_synset_encoder(embedding)(corpus, word2vec_model, rootpath=rootpath)
    makedirsforfile(resfile)

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
        printStatus(INFO, '%d done, %d okay' % ((i+1), len(good)))
 

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
    parser = OptionParser(usage="""usage: %prog [options] synset_file synset_name""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)
    parser.add_option("--embedding", default=DEFAULT_EMBEDDING, type="string", help="embedding model (default: %s)" % DEFAULT_EMBEDDING)
    
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
