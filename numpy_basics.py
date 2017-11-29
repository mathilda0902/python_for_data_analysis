'''numpy'''
'''np.ones_like(arr):
np.zeros_like(arr):
produce array of ones with same length and dtype.'''
a = np.array(['j','a','g','y'])
np.ones_like(a)
'''Out[7]:
array(['1', '1', '1', '1'],
      dtype='|S1')'''

'''np.eye(n) = np.identity(n)'''

arr[n][m] '''equiv to''' arr[n, m]

'''fancy indexing'''
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr[[4,3,0,6]]
arr[[-3, -5, -7]] #selecting from the last row

arr = np.arange(32).reshape((8,4))
arr[[1,5,7,2], [0,3,1,2]]
'''Out[17]: array([ 4, 23, 29, 10])''' #[A], [B]: row indeces A, column indeces B

arr[np.ix_([1,5,7,2], [0,3,1,2])] #converts two 1D integer arrays to an indexer that selects the square region

arr = np.arange(16).reshape(2,2,4)
arr.swapaxes(1,2) #changing view but not changing array

'''unary ufuncs: fast element-wise array functions:'''
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)

arr = np.random.randn(7) * 5
np.modf(arr) # returns the fractional and integral parts of a floating point array:
'''
In [33]: arr
Out[33]:
array([-6.68532786,  2.40788553, -7.02318104,  2.40351574,  0.31961334,
        1.6051586 ,  8.74808986])

In [34]: np.modf(arr)
Out[34]:
(array([-0.68532786,  0.40788553, -0.02318104,  0.40351574,  0.31961334,
         0.6051586 ,  0.74808986]), array([-6.,  2., -7.,  2.,  0.,  1.,  8.]))
'''

'''
np.argmin, np.argmax: returns the first index of min, max
np.fabs equiv to np.abs
np.log1p: log(1+x)
np.log: ln
np.sign: 1, 0, -1 depending on positivity/zero/negativity
np.rint: round to the nearest integer
np.isnan: is NaN?
np.isinf: is infinite?
np.logical_not: not x elementwise
np.fmax, np.fmin: element-wise max, min, ignores NaN
np.mod: element-wise modulus
np.copysign: copy sign of values in second argument to values in first argument
'''

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
'''
In [56]: ys
Out[56]:
array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
       [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
       [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
       ...,
       [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
       [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
       [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])

In [57]: xs
Out[57]:
array([[-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
       ...,
       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99]])
'''

'''
np.where(cond, xarr, yarr): xarr and yarr can either or both be scalars
                            cond: boolean condition

In [64]: arr=np.random.randn(4,4)
In [65]: arr
Out[65]:
array([[ 1.24320663,  1.57181735, -0.07014987, -1.0581978 ],
       [ 0.72183718, -0.79145817, -0.81255215,  1.11328337],
       [ 1.67870931, -0.54043587, -0.27528524, -2.06264459],
       [-1.85535897, -0.07440977, -0.66986888, -0.16907991]])

In [66]: np.where(arr>0, 1, -1)
Out[66]:
array([[ 1,  1, -1, -1],
       [ 1, -1, -1,  1],
       [ 1, -1, -1, -1],
       [-1, -1, -1, -1]])
In [67]: np.where(arr>0, 1, arr)
Out[67]:
array([[ 1.        ,  1.        , -0.07014987, -1.0581978 ],
       [ 1.        , -0.79145817, -0.81255215,  1.        ],
       [ 1.        , -0.54043587, -0.27528524, -2.06264459],
       [-1.85535897, -0.07440977, -0.66986888, -0.16907991]])
'''

'''eg:
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

equiv to:

np.where(cond1 & cond2, 0,
        np.where(cond1, 1,
                np.where(cond2, 2, 3)))
'''

'''basic array statistical methods:
np.argmin, np.argmax: indeces of min and max elements
np.cumsum(axis): cumulative sum of elements starting from 0
np.cumprod(axis): cumulative product of elements starting from 1
axis = 0, 1: by row, column
'''

'''use np.sum to count boolean:'''
arr=np.random.randn(100)
(arr > 0).sum()

'''.any(), .all(): works for both boolean and non-boolean arrays
                    non-zero elements evaluated to be true
.any(): returns True if any is non-zero
.all(): returns True if all are non-zero
'''

'''
arr.sort(): sort in place
'''

'''compute sample 5% percentile:'''
large_arr = np.random.randn(1000)
large_arr.sort(ascending=False)
large_arr[int(0.05 * len(large_arr))]
'''Out[101]: -1.5805147556436223'''

'''order by keyword, sort:'''
dtype = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38), ('Galahad', 1.7, 38)]
a = np.array(values, dtype=dtype)       # create a structured array
np.sort(a, order='height')
'''array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
       ('Lancelot', 1.8999999999999999, 38)],
      dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])'''

'''Sort by age, then height if ages are equal:'''
np.sort(a, order=['age', 'height'])
'''array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
       ('Arthur', 1.8, 41)],
      dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])'''

'''
np.unique(arr): returns unique values
                same as:
                sprted(set(names))
np.in1d(arr1, arr2):    arr1 and arr2 both are 1D arrays.
                        returns boolean array of len(arr1),
                        True if element of arr1 in arr2
                        False if element of arr1 not in arr2
np.intersect1d(arr1, arr2): sorted, common elements in both
np.setdiff1d(arr1, arr2): elements in arr1 that are not in arr2
np.setxor1d(arr1, arr2): elements in each but not both
'''

'''np.savetxt and np.loadtxt'''
np.savetxt('random_numbers.txt', x, delimiter='-', newline='\n',
            header="Beginning of file.",
            footer="These numbers were generated randomly.")
arr = np.loadtxt('random_numbers.txt', delimiter='x')

'''numpy.linalg:
np.dot(x, y): matrix multiplication
np.diag:    1) returns diagonal elements of a square matrix, as 1D array, or
            2) converts a 1D array into a diagonal square matrix
np.trace: returns sum of diagonal elements of a square matrix
np.det: compute determinant
np.eig: eigenvalues and eigenvectors of a square matrix
np.inv: inverse of a square matrix
np.pinv: MP pseudo-inverse inverse of a matrix
np.qr: QR decomposition
np.svd: singular value decomposition
np.solve: solve Ax = b for x, A a square matrix
np.lstsq: least-squares solution to Ax = b
'''
from numpy.linalg import inv, qr

'''numpy.random functions:
np.random.seed:        seed random number generator
np.random.permutation: returns a random permutation of a sequence
np.random.shuffle:     randomly permutes a sequence IN PLACE
np.random.rand:        draw samples from a uniform distribution
np.random.randint:     draw integers from a given low-to-high range
np.random.randn:       draw samples from standard normal distribution
np.random.binomial:    draw samples from a binomial distribution
np.random.normal:      draw samples from a normal(Gaussian) distribution
np.random.beta:        draw samples from a beta distribution
np.random.chisquare:   draw samples from a chi-square distribution
np.random.gamma:       draw samples from a gamma distribution
np.random.uniform:     draw samples from a uniform [0, 1) distribution
'''

'''random walk:'''
import random
np.seed=10
steps = 1000
position = 0
walk = [position]
for i in xrange(steps):
    step = 1 if np.random.randint(0, 2) else -1
    position += step
    walk.append(position)
print len(walk)

'''np way:'''
np.seed=10
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
print len(walk)

'''many random walks at once'''
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
