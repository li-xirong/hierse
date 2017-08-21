from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

testCollection = 'imagenet2hop-random2k'

for embedding in str.split('conse conse2 hierse hierse2'):
    method_name = os.path.join('dascaffeprob', 'flickr4m,tagvec500,%s' % embedding)
    os.system('python zero_shot_tagging.py %s --embedding %s --overwrite 0' % (testCollection, embedding))
    os.system('python evaluate.py %s %s' % (testCollection, method_name))


