import os

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
DEFAULT_WNID2WORDS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/wnid2words.pkl')
DEFAULT_IS_A_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/wordnet.is_a.txt')

DEFAULT_W2V_CORPUS = 'flickr4m'
DEFAULT_W2V = 'tagvec500'
DEFAULT_EMBEDDING = 'hierse2'

DEFAULT_Y0 = 'ilsvrc12_test1k'
DEFAULT_Y1 = 'ilsvrc12_test1k_2hop'
DEFAULT_pY0 = 'dascaffeprob'

#DEFAULT_Y0_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/synsets_%s.txt' % DEFAULT_Y0)
DEFAULT_LABEL_VEC_NAME = '%s,%s,%s' % (DEFAULT_W2V_CORPUS, DEFAULT_W2V, DEFAULT_EMBEDDING)

DEFAULT_MAX_LAYER = 7
DEFAULT_K = 10
DEFAULT_BATCH_SIZE = 2000