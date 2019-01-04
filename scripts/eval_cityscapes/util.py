# The following code is modified from https://github.com/shelhamer/clockwork-fcn
import numpy as np
import scipy.io as sio

def get_out_scoremap(net):
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net)

def fast_hist(a, b, n):
    # print('saving')
    # sio.savemat('/tmp/fcn_debug/xx.mat', {'a':a, 'b':b, 'n':n})
    k = np.where((a >= 0) & (a < n))[0]
   # both_13 = np.where((a==13) & (b==13))
   # print( both_13 )
    bin_array = n * a[k].astype(int) + b[k].astype(int)
    m = np.where( bin_array > 360 )
    bin_array[m] = 359
    #print(np.max(bin_array))
    #print(a[np.where(a[np.where(a>12)]!=255)],b[np.where(b[np.where(b>12)]!=255)])
    bc = np.bincount(bin_array, minlength=361)
    #print('diagonal12-15', bc[12*19+12],bc[13*19+13],bc[14*19+14],bc[15*19+15])
    #print(a[k].dtype, b[k].dtype)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)

def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
    print(hist.sum())
    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu
