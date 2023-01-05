from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='30000', type=int) # our experiment iteration 30000
parser.add_argument('--batch_size', default='1', type=int)
parser.add_argument('--cache', default=False, type=boolean_string)
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf['data'])
print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# outdoor1, outdoor2, diff cloth, indoor
for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print('s1: %.3f,\ts2: %.3f,\ts3: %.3f,\ts4: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i]),
        np.mean(acc[3, :, :, i])))

# for i in range(1):
#     print('===Rank-%d (Include identical-view cases)===' % (i + 1))
#     print('normal: %.3f,\tpacket: %.3f,\tbackpack: %.3f,\tgown: %.3f' % (
#         np.mean(acc[0, :, :, i]),
#         np.mean(acc[1, :, :, i]),
#         np.mean(acc[2, :, :, i]),
#         np.mean(acc[3, :, :, i])))