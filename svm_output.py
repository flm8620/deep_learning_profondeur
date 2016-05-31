import sys
import os
import time
import argparse

from sklearn import svm as svm
from sklearn.decomposition import PCA
import six.moves.cPickle as pickle
import numpy
import functools


def log_and_print(text, logfile):
    with open(logfile, 'a') as f:
        f.write(text + '\n')
        print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file")
    parser.add_argument('-p', '--pca', help="use PCA", action='store_true')
    parser.add_argument('-n', '--n-pca', help="dim of PCA", default=100)

    args = parser.parse_args()

    datafile = args.datafile
    usePCA = args.pca
    nPCA = args.n_pca
    save_file_name = 'svm_' + datafile + '.txt'
    logfile = save_file_name + '_log.txt'
    log_print = functools.partial(log_and_print, logfile=logfile)

    log_print('--Parameters--')
    log_print('  use PCA : {}'.format(usePCA))
    if usePCA:
        log_print('    dim PCA = {}'.format(nPCA))
    print()

    log_print('loading ' + datafile + ' ... ')
    with open(datafile, 'rb') as f:
        [all_train_output, all_train_y, all_test_output, all_test_y] = pickle.load(f)

    assert len(all_train_output) == len(all_train_y)
    assert len(all_test_output) == len(all_test_y)
    log_print('{} training examples'.format(len(all_train_y)))
    log_print('{} testing examples'.format(len(all_test_y)))
    log_print('data dimension : {}'.format(len(all_train_output[0])))

    if usePCA:
        log_print('fitting PCA with ndim = {}'.format(nPCA))
        pca = PCA(n_components=nPCA)
        pca.fit(all_train_output)
        all_train_output = pca.transform(all_train_output)
        all_test_output = pca.transform(all_test_output)
    else:
        log_print('not using PCA')

    lsvm = svm.LinearSVC()

    log_print('fitting Linear SVM ...')
    lsvm.fit(all_train_output, all_train_y)

    log_print('testing Linear SVM ...')
    pre = lsvm.predict(all_test_output)

    wrong = 0
    for i in range(len(pre)):
        if pre[i] != all_test_y[i]:
            wrong += 1

        log_print('test acc : {}'.format(wrong / float(len(pre))))


if __name__ == '__main__':
    main()
