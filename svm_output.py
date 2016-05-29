import sys
import os
import time
import argparse

from sklearn import svm as svm
from sklearn.decomposition import PCA
import six.moves.cPickle as pickle
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file")
    parser.add_argument('-p', '--pca', help="use PCA", action='store_true')
    parser.add_argument('-n', '--n-pca', help="dim of PCA", default=100)

    args = parser.parse_args()

    datafile = args.datafile
    usePCA = args.pca
    nPCA = args.n_pca

    print('--Parameters--')
    print('  use PCA : ', usePCA)
    if usePCA:
        print('    dim PCA = ', nPCA)
    print()

    print('loading ' + datafile + ' ... ')
    with open(datafile, 'rb') as f:
        [all_train_output, all_train_y, all_test_output, all_test_y] = pickle.load(f)

    assert len(all_train_output) == len(all_train_y)
    assert len(all_test_output) == len(all_test_y)
    print('{} training examples'.format(len(all_train_y)))
    print('{} testing examples'.format(len(all_test_y)))
    print('data dimension : {}'.format(len(all_train_output[0])))

    if usePCA:
        print('fitting PCA with ndim = {}'.format(nPCA))
        pca = PCA(n_components=nPCA)
        pca.fit(all_train_output)
        all_train_output = pca.transform(all_train_output)
        all_test_output = pca.transform(all_test_output)
    else:
        print('not using PCA')

    lsvm = svm.LinearSVC()

    print('fitting Linear SVM ...')
    lsvm.fit(all_train_output, all_train_y)

    print('testing Linear SVM ...')
    pre = lsvm.predict(all_test_output)

    wrong = 0
    for i in range(len(pre)):
        if pre[i] != all_test_y[i]:
            wrong += 1

    print('test acc :', wrong / float(len(pre)))


if __name__ == '__main__':
    main()
