import sys
import os
import numpy as np
from basic.common import checkToSkip, makedirsforfile, niceNumber, ROOT_PATH, printStatus
from basic.util import readImageSet
from simpleknn.bigfile import BigFile

from tagger import ZeroshotTagger
from im2vec import Image2Vec
from build_synset_vec import DEFAULT_CORPUS, DEFAULT_WORD2VEC, DEFAULT_EMBEDDING

DEFAULT_Y0 = 'imagenet1k'
DEFAULT_Y1 = 'imagenet1k2hop'
DEFAULT_pY0 = 'dascaffeprob'
DEFAULT_R = 20

INFO = os.path.basename(__file__)


def process(options, testCollection):
    overwrite = options.overwrite
    rootpath = options.rootpath
    corpus = options.corpus
    word2vec_model = options.word2vec
    embedding_model = options.embedding
    Y0 = options.Y0
    Y1 = options.Y1
    pY0 = options.pY0
    r = options.r
    blocksize = 2000

    embedding_name = '%s,%s,%s' % (corpus, word2vec_model, embedding_model)
    for synset_name in [Y0, Y1]:
        assert(os.path.exists(os.path.join(rootpath, 'synset2vec', synset_name, embedding_name)))

    resfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, embedding_name, pY0, 'id.tagvotes.txt')
    if checkToSkip(resfile, overwrite):
        return 0

    label_file = 'data/ilsvrc12/synsets.txt'
    label2vec_dir = os.path.join(rootpath, 'synset2vec', Y0, embedding_name)
    i2v = Image2Vec(label_file, label2vec_dir)

    tagger = ZeroshotTagger(synset_name=Y1, embedding_name=embedding_name, rootpath=rootpath)

    imset = readImageSet(testCollection, testCollection, rootpath)
    feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', pY0)
    feat_file = BigFile(feat_dir)
    

    printStatus(INFO, 'tagging %d images' % len(imset))
    makedirsforfile(resfile)
    fw = open(resfile, 'w')

    start = 0
    while start < len(imset):
        end = min(len(imset), start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end))
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
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)
    parser.add_option("--embedding", default=DEFAULT_EMBEDDING, type="string", help="embedding model (default: %s)" % DEFAULT_EMBEDDING)
    parser.add_option("--Y0", default=DEFAULT_Y0, type="string", help="training labels (default: %s)" % DEFAULT_Y0)
    parser.add_option("--Y1", default=DEFAULT_Y1, type="string", help="test labels (default: %s)" % DEFAULT_Y1)
    parser.add_option("--pY0", default=DEFAULT_pY0, type="string", help="probabilistic prediction of Y0 (default: %s)" % DEFAULT_pY0)
    parser.add_option("--r", default=DEFAULT_R, type="int", help="how many tags to predict (default: %d)" % DEFAULT_R)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    assert('imagenet1k' == options.Y0)
    assert('imagenet1k2hop' == options.Y1)
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())
