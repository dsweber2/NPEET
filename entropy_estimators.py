#!/usr/bin/env python
# Written by Greg Ver Steeg
# See readme.pdf for documentation
# Or go to http://www.isi.edu/~gregv/npeet.html

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np
import random
from numpy import linalg as la
try:
    import multiprocessing as processing
except:
    import processing




def __remote_process_query(rank, qin, qout, tree, K, leafsize):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        # search for the distance to the k nearest points; the blank is their locations
        knn,_ = tree.query(data, K, p=float('inf'))
        # write to output queue
        qout.put((nc,knn))

def __remote_process_ball(rank, qin, qout, tree, leafsize):
    while 1:
        # read input queue (block until data arrives)
        nc, data, eps = qin.get()
        assert len(eps) == data.shape[0]
        knn = []
        #listOfPoints = tree.query_ball_point(data, eps, p=float('inf'))
        for i in range(len(eps)):
            dist = eps[i]-1e-15
            knn += [len(tree.query_ball_point(data[i,:], dist, p=float('inf')))+1]
        # write to output queue
        qout.put((nc,knn))

def knn_search_parallel(data, K, qin=None, qout=None, tree=None, t0=None, eps=None, leafsize=None, copy_data=False):
    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree, exploiting all logical
        processors on the computer. if eps <= 0, it returns the distance to the kth point. On the other hand, if eps > 0 """
    # print("starting the parallel search")
    if eps is not None:
        assert data.shape[0]==len(eps)
    # build kdtree
    if copy_data:
        dataCopy = data.copy()
        # print('copied data')
    else:
        dataCopy = data
    if tree is None and leafsize is None:
        tree = ss.cKDTree(dataCopy)
    elif tree is None:
        tree = ss.cKDTree(dataCopy, leafsize=leafsize)
    if t0 is not None:
        print('time to tree formation: %f' %(clock()-t0))
    ndata = data.shape[0]
    nproc = 20
    # print('made the tree')
    # compute chunk size
    chunk_size = int(data.shape[0] / (4*nproc))
    chunk_size = 100 if chunk_size < 100 else chunk_size
    if qin==None or qout==None:
        # set up a pool of processes
        qin = processing.Queue(maxsize=int(ndata/chunk_size))
        qout = processing.Queue(maxsize=int(ndata/chunk_size))
    if eps is None:
        pool = [processing.Process(target=__remote_process_query,
                                   args=(rank, qin, qout, tree, K, leafsize))
                for rank in range(nproc)]
    else:
        pool = [processing.Process(target=__remote_process_ball,
                                   args=(rank, qin, qout, tree, leafsize))
                for rank in range(nproc)]
    for p in pool: p.start()
    # put data chunks in input queue
    cur, nc = 0, 0
    while 1:
        _data = data[cur:cur+chunk_size, :]
        if _data.shape[0] == 0: break
        if eps is None:
            qin.put((nc,_data))
        else:
            _eps = eps[cur:cur+chunk_size]
            qin.put((nc,_data,_eps))
        cur += chunk_size
        nc += 1
    # read output queue
    knn = []
    while len(knn) < nc:
        knn += [qout.get()]
        # avoid race condition
        _knn = [n for i,n in sorted(knn)]
    knn = []
    for tmp in _knn:
        knn += [tmp]
    # terminate workers
    for p in pool: p.terminate()

    if eps is None:
        output = np.zeros((sum([ x.shape[0] for x in knn]),knn[0].shape[1]))
    else:
        output = np.zeros(sum([ len(x) for x in knn]))
    outputi = 0
    for x in knn:
        if eps is None:
            nextVal = x.shape[0]
        else:
            nextVal = len(x)
        output[outputi:(outputi+nextVal)] = x
        outputi += nextVal
    return output

# CONTINUOUS ESTIMATORS

def entropy(x, k=3, base=2.0,printing=False, qin=None, qout=None):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = x + intens*nr.rand(x.shape[0],x.shape[1])
    nn = knn_search_parallel(x,k,qin=qin, qout=qout)[:,-1]
    #tree = ss.ccKDTree(x)
    # [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
    const = digamma(N) - digamma(k) + d * log(2)
    h = (const + d * np.mean(np.log(nn))) / log(base)
    if printing:
        print(h)
    return h

def centropy(x, y, k=3, base=2):
  """ The classic K-L k-nearest neighbor continuous entropy estimator for the
      entropy of X conditioned on Y.
  """
  hxy = entropy([xi + yi for (xi, yi) in zip(x, y)], k, base)
  hy = entropy(y, k, base)
  return hxy - hy

def column(xs, i):
  return [[x[i]] for x in xs]

def tc(xs, k=3, base=2):
  xis = [entropy(column(xs, i), k, base) for i in range(0, len(xs[0]))]
  return np.sum(xis) - entropy(xs, k, base)

def ctc(xs, y, k=3, base=2):
  xis = [centropy(column(xs, i), y, k, base) for i in range(0, len(xs[0]))]
  return np.sum(xis) - centropy(xs, y, k, base)

def corex(xs, ys, k=3, base=2):
  cxis = [mi(column(xs, i), ys, k, base) for i in range(0, len(xs[0]))]
  return np.sum(cxis) - mi(xs, ys, k, base)

def mi(xp, yp, k=3, base=2,normalize=True,qin=None, qout=None):
    """ Mutual information of x and y
        x, y should eiter be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]] or a matrix with (#examples x dimension). Their types should match
    """
    assert xp.shape[0] == yp.shape[0], "Lists should have same length"
    assert type(xp[0])==type(yp[0]), "Should be the same kind of list"
    assert k <= len(xp) - 1, "Set k smaller than num. samples - 1. Way smaller."
    x = (xp-np.mean(xp,axis=0))/np.std(xp, axis=0)
    y = (yp-np.mean(yp,axis=0))/np.std(yp, axis=0)
    intens = 1e-10
    points = np.append(x,y,axis=1) + intens*nr.rand(x.shape[0],x.shape[1]+y.shape[1])
    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = knn_search_parallel(points, k+1, qin=qin, qout=qout)
    dvec = dvec[:,-1]
    #tree = ss.ccKDTree(points)
    #dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(x, dvec,qin=qin, qout=qout), avgdigamma(y, dvec, qin=qin, qout=qout), digamma(k), digamma(points.shape[0])
    mmi = (-a - b + c + d) / log(base)
    return mmi


def cmi(x, y, z, k=3, base=2, qin=None, qout=None):
    """ Mutual information of x and y, conditioned on z
        x, y, z should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
    z = [list(p + intens * nr.rand(len(z[0]))) for p in z]
    points = zip2(x, y, z)
    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = knn_search_parallel(points,k,qin=qin, qout=qout)[:,-1]
    #tree = ss.ccKDTree(points)
    #[tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(zip2(x, z), dvec,qin=qin, qout=qout), avgdigamma(zip2(y, z), dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def kldiv(x, xp, k=3, base=2, qin=None, qout=None):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
        x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    assert k <= len(xp) - 1, "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n - 1)
    tree = ss.ccKDTree(x)
    treep = ss.ccKDTree(xp)
    nn = knn_search_parallel(x, k, tree=tree, qin=qin, qout=qout)
    #[tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
    nnp = knn_search_parallel(x, k-1, tree=treep, qin=qin, qout=qout)
    #nnp = [treep.query(point, k, p=float('inf'))[0][k - 1] for point in x]
    return (const + d * np.mean(map(log, nnp)) - d * np.mean(map(log, nn))) / log(base)


# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    """ Discrete entropy estimator
        Given a list of samples which can be any hashable object
    """
    return entropyfromprobs(hist(sx), base=base)


def midd(x, y, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    return -entropyd(zip(x, y), base) + entropyd(x, base) + entropyd(y, base)

def cmidd(x, y, z):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    return entropyd(zip(y, z)) + entropyd(zip(x, z)) - entropyd(zip(x, y, z)) - entropyd(z)

def centropyd(x, y, base=2):
  """ The classic K-L k-nearest neighbor continuous entropy estimator for the
      entropy of X conditioned on Y.
  """
  return entropyd(zip(x, y), base) - entropyd(y, base)

def tcd(xs, base=2):
  xis = [entropyd(column(xs, i), base) for i in range(0, len(xs[0]))]
  hx = entropyd(xs, base)
  return np.sum(xis) - hx

def ctcd(xs, y, base=2):
  xis = [centropyd(column(xs, i), y, base) for i in range(0, len(xs[0]))]
  return np.sum(xis) - centropyd(xs, y, base)

def corexd(xs, ys, base=2):
  cxis = [midd(column(xs, i), ys, base) for i in range(0, len(xs[0]))]
  return np.sum(cxis) - midd(xs, ys, base)

def hist(sx):
    sx = discretize(sx)
    # Histogram from list of samples
    d = dict()
    for s in sx:
        if type(s) == list:
          s = tuple(s)
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z) / len(sx), d.values())


def entropyfromprobs(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs)) / log(base)


def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x * log(x)
tmp = [x for x in range(20)]
# MIXED ESTIMATORS
def micd(x, y, k=3, base=2, warning=True, qin=None,qout=None):
    """ If x is continuous and y is discrete, compute mutual information
    """
    overallentropy = entropy(x, k, base, qin=qin,qout=qout)

    if type(y)==np.ndarray:
        y = y.tolist()
    n = len(y)
    word_dict = dict()
    for i in range(len(y)):
      if type(y[i]) == list:
        y[i] = tuple(y[i])
    for sample in y:
        word_dict[sample] = word_dict.get(sample, 0) + 1. / n
    yvals = list(set(word_dict.keys()))

    mi = overallentropy
    for yval in yvals:
        xgiveny = x[[k==1 for k in y],:]
        if k <= len(xgiveny) - 1:
            mi -= word_dict[yval] * entropy(xgiveny, k, base,qin=qin,qout=qout)
        else:
            if warning:
                print("Warning, after conditioning, on y=", yval, " insufficient data. Assuming maximal entropy in this case.")
            mi -= word_dict[yval] * overallentropy
    print(np.abs(mi))
    return np.abs(mi)  # units already applied

def midc(x, y, k=3, base=2, warning=True, qin=None,qout=None):
  return micd(y, x, k, base, warning)

def centropydc(x, y, k=3, base=2, warning=True):
  return entropyd(x, base) - midc(x, y, k, base, warning)

def centropycd(x, y, k=3, base=2, warning=True):
  return entropy(x, k, base) - micd(x, y, k, base, warning)

def ctcdc(xs, y, k=3, base=2, warning=True):
  xis = [centropydc(column(xs, i), y, k, base, warning) for i in range(0, len(xs[0]))]
  return np.sum(xis) - centropydc(xs, y, k, base, warning)

def ctccd(xs, y, k=3, base=2, warning=True):
  xis = [centropycd(column(xs, i), y, k, base, warning) for i in range(0, len(xs[0]))]
  return np.sum(xis) - centropycd(xs, y, k, base, warning)

def corexcd(xs, ys, k=3, base=2, warning=True):
  cxis = [micd(column(xs, i), ys, k, base, warning) for i in range(0, len(xs[0]))]
  return np.sum(cxis) - micd(xs, ys, k, base, warning)

def corexdc(xs, ys, k=3, base=2, warning=True):
  #cxis = [midc(column(xs, i), ys, k, base, warning) for i in range(0, len(xs[0]))]
  #joint = midc(xs, ys, k, base, warning)
  #return np.sum(cxis) - joint
  return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)

# UTILITY FUNCTIONS
def vectorize(scalarlist):
    """ Turn a list of scalars into a list of one-d vectors
    """
    return [[x] for x in scalarlist]


def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """ Shuffle test
        Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
        Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
        'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
        Mutual information and CMI should have a mean near zero.
    """
    xp = x[:]  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        random.shuffle(xp)
        if z:
            outputs.append(measure(xp, y, z, **kwargs))
        else:
            outputs.append(measure(xp, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])


# INTERNAL FUNCTIONS

def avgdigamma(points, dvec, qin=None, qout=None):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    N = points.shape[0]
    avg = 0.
    num_points = knn_search_parallel(points, 3, eps=dvec-1e-15, qin=qin, qout=qout)
    avg = sum([digamma(x)/N for x in num_points])
    #tree = ss.ccKDTree(points)

    #for i in range(N):
    #dist = dvec[i]
    # subtlety, we don't include the boundary point,
    # but we are implicitly adding 1 to kraskov def bc center point is included

    #num_points = len(tree.query_ball_point(points[i,:], dist - 1e-15, p=float('inf')))
    #avg += digamma(num_points) / N
    return avg


def zip2(*args):
    # zip2(x, y) takes the lists of vectors and makes it a list of vectors in a joint space
    # E.g. zip2([[1], [2], [3]], [[4], [5], [6]]) = [[1, 4], [2, 5], [3, 6]]
    return [sum(sublist, []) for sublist in zip(*args)]

def discretize(xs):
    def discretize_one(x):
        if len(x) > 1:
            return tuple(x)
        else:
            return x[0]
    # discretize(xs) takes a list of vectors and makes it a list of tuples or scalars
    return [discretize_one(x) for x in xs]

#if __name__ == "__main__":
#    print("NPEET: Non-parametric entropy estimation toolbox. See readme.pdf for details on usage.")

from time import clock
from numpy import random as rand
def test(t0):
    return knn_search_parallel(data, K,t0=t0)

if __name__ == '__main__':
    dY, dX = (3072,3072)
    ndata = 6000
    A = np.block([[10*nr.randn(dX,dX), nr.randn(dX,dY)], [nr.randn(dY,dX), 10*nr.randn(dY,dY)]])
    A = A.T@A
    AX = A[0:dX,0:dX]
    AY = A[dX:,dX:]
    data = 10*nr.multivariate_normal(np.zeros(dX+dY), A, ndata)
    eps = 3*nr.rand(ndata)
    ss.cKDTree(data.copy())
    #print('avgdigamma with random eps: %f' % test3)
    print('Testing mutual information calculation:')
    mi(data[:,0:dX],data[:,dX:])
    print('Theoretical Value')
    print((la.slogdet(AX)[1]+la.slogdet(AY)[1]-la.slogdet(A)[1])/2)
