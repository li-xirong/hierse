# coding: utf-8
import sys
import os
import time
import numpy as np
from basic.common import checkToSkip, makedirsforfile, niceNumber, ROOT_PATH, printStatus
from simpleknn.bigfile import BigFile


DEFAULT_K = 10
DEFAULT_BLOCK_SIZE = 2000

INFO = os.path.basename(__file__)

class Image2Vec:

    def __init__(self, label_file, label2vec_dir):
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
        printStatus(INFO, '#active_labels=%d, embedding_size=%d' % (self.nr_of_labels - nr_of_inactive_labels, self.feat_dim))



    def embedding(self, prob_vec, k=10):
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




def process(options, label_file, label2vec_dir, testCollection, feature, new_feature):
    rootpath = options.rootpath
    overwrite = options.overwrite
    k = options.k
    blocksize = options.blocksize
    subset = options.subset if options.subset else testCollection

    resfile = os.path.join(rootpath, testCollection, 'FeatureData', new_feature, 'id.feature.txt')
    if checkToSkip(resfile, overwrite):
        return 0

    imsetfile = os.path.join(rootpath, testCollection, 'ImageSets', '%s.txt' % subset)
    imset = map(str.strip, open(imsetfile).readlines())
    printStatus(INFO, '%d images to do' % len(imset))

    feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', feature))

    im2vec = Image2Vec(label_file, label2vec_dir)


    makedirsforfile(resfile)
    fw = open(resfile, 'w')

    read_time = 0
    run_time = 0
    start = 0
    done = 0

    while start < len(imset):
        end = min(len(imset), start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed, test_X = feat_file.read(imset[start:end])
        read_time += time.time() - s_time
        
        s_time = time.time()
        output = [None] * len(renamed)
        for i in xrange(len(renamed)):
            vec = im2vec.embedding(test_X[i], k)
            output[i] = '%s %s\n' % (renamed[i], " ".join([niceNumber(x,6) for x in vec]))
        run_time += time.time() - s_time
        start = end
        fw.write(''.join(output))
        done += len(output)

    # done    
    printStatus(INFO, "%d done. read time %g seconds, run_time %g seconds" % (done, read_time, run_time))
    fw.close()
    return done


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] label_file label2vec_dir testCollection feature new_feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--subset", default="", type="string", help="only do this subset")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="top-k labels used for semantic embedding (default: %d)" % DEFAULT_K)
    parser.add_option("--blocksize", default=DEFAULT_BLOCK_SIZE, type="int", help="nr of feature vectors loaded into memory (default: %d)" % DEFAULT_BLOCK_SIZE)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3], args[4])


if __name__ == "__main__":
    sys.exit(main())
