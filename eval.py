# coding: utf8
import os
import sys
from basic.common import ROOT_PATH, printStatus

INFO = os.path.basename(__file__)


class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length>len(sorted_labels) or length<=0:
            length = len(sorted_labels)
        return length

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")


class HitScorer (MetricScorer):

    def score(self, sorted_labels):
        length = self.getLength(sorted_labels)
        for i in xrange(length):
            if 1 <= sorted_labels[i]:
                return 1.0
        return 0.0



# For Imagenet, the ground truth synset ID is included in the imagenetID:  {wnid}_{suffix}
def load_ground_truth(collection, imset=None, rootpath=ROOT_PATH):
    if not imset:
        imset = map(str.strip, open(os.path.join(rootpath, collection, 'ImageSets', '%s.txt'%collection)).readlines())
    im2truth = dict([(x, set( [x.split("_")[0] ] )) for x in imset])
    return im2truth



def process(options, testCollection, method):
    rootpath = options.rootpath

    scorers = [HitScorer(k) for k in [1, 2, 5, 10]]
    im2truth = load_ground_truth(testCollection, imset=None, rootpath=rootpath)
    printStatus(INFO, 'nr of ground-truthed images: %d' % len(im2truth))

    tag_prediction_file = os.path.join(rootpath, testCollection,'autotagging', testCollection, method, 'id.tagvotes.txt')
    printStatus(INFO, 'evaluating %s' % tag_prediction_file)
    res = [0] * len(scorers)
    nr_of_images = 0

    for line in open(tag_prediction_file):
        elems = line.strip().split()
        imageid = elems[0]
        del elems[0]
        assert(len(elems)%2 == 0)
        pred_labels = [elems[i] for i in range(0, len(elems), 2)]
        pred_labels = pred_labels[:10] # consider at most the first 20 predicted tags
        truth = im2truth.get(imageid, None)
        if not truth:
            continue
        sorted_labels = [int(x in truth) for x in pred_labels]
        perf = [scorer.score(sorted_labels) for scorer in scorers]
        res = [res[i] + perf[i] for i in range(len(scorers))]
        nr_of_images += 1

    printStatus(INFO, 'nr of images: %d' % nr_of_images)
    res = [x/nr_of_images for x in res]

    print ' '.join([x.name() for x in scorers])
    print ' '.join(['%.3f' % x for x in res])



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser

    parser = OptionParser(usage="""usage: %prog [options] testCollection method""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--metric", type="string", default="hit",  help="performance metric, namely hit")
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    assert (options.metric in ['hit'])   

    return process(options, args[0], args[1])



if __name__=="__main__":
    sys.exit(main())

