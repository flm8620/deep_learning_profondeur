import six.moves.cPickle as pickle

def get_cifar10():
    file = 'cifar-10-batches-py/data_batch_1'
    fo = open(file, 'rb')
    var = pickle.load(fo,encoding='latin1')
    fo.close()
    return var 
