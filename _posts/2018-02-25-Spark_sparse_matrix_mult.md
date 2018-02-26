---
layout: post
mathjax: true
title: Sparse matrix-vector multiplication in Spark
tags: [sparse matrices, spark, linear algebra, numpy]
---

_Sparse matrix multiplication using Spark RDDs._

### Sparse matrices

Sparse matrices are defined as matrices in which most elements are zero. Specifically, the sparsity of a matrix is defined as

$$
\frac{\text{number of zero-valued elements}}{\text{total number of elements}}.
$$

Sparse matrices describe loosely coupled linear systems. It is often convenient to store sparse matrices in [COO (coordinate list)](https://en.wikipedia.org/wiki/Sparse_matrix#Storing_a_sparse_matrix) format. This allows us to define only the non-zero elements of the matrix as a list of 3-tuples: $(i, j, v)$, such that $M_{ij}=v$. As an example, here's some Python code that uses NumPy to generate a random, sparse matrix in $\mathbf{R}^{\text{10,000}\times \text{10,000}}$ with 20,000 non-zero entries between 0 and 1. We'll also make use of the `coo_matrix` class from `scipy.sparse`, which allows us to quickly convert to a dense format for testing.


{% highlight python%}
import numpy as np
from scipy.sparse import coo_matrix
from pyspark import SparkConf, SparkContext
{% endhighlight %}


{% highlight python%}
n = 10000
{% endhighlight %}


{% highlight python%}
indices = np.random.randint(0, n, size=(2*n, 2))
values = np.random.random(size=2*n)
{% endhighlight %}


{% highlight python%}
sparse_representation = np.c_[indices, values[:, None]]
{% endhighlight %}


{% highlight python%}
sparse_representation[:5]
{% endhighlight %}



{% highlight python%}
    array([[6.86000000e+02, 3.38500000e+03, 7.94401577e-01],
           [5.86500000e+03, 5.35100000e+03, 7.74288349e-01],
           [1.59000000e+03, 5.72300000e+03, 3.41039090e-01],
           [1.31100000e+03, 9.25600000e+03, 3.44232609e-01],
           [9.03100000e+03, 4.97900000e+03, 9.57372493e-01]])
{% endhighlight %}


We'll save this to disk for future use.


{% highlight python%}
np.savetxt('sparse_matrix.txt', sparse_representation, delimiter=' ')
{% endhighlight %}

The `coo_matrix` class constructs a sparse matrix using the form `(data, (i, j)`, where `data`, `i`, and `j` are arrays:


1. `data[:]`, the entries of the matrix, in any order
2. `i[:]`, the row indices of the matrix entries
3. `j[:]`, the column indices of the matrix entries

The SciPy [sparse matrix formats](https://docs.scipy.org/doc/scipy/reference/sparse.html) are super useful and are compatible with [sklearn algorithms](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html). Here, we'll just use it to convert our sparse representation to a dense array for comparison and testing.


{% highlight python%}
M_sparse = coo_matrix((values, (indices.T[0], indices.T[1])), shape=(n, n))
M_sparse
{% endhighlight %}



{% highlight python%}
    <10000x10000 sparse matrix of type '<type 'numpy.float64'>'
    	with 20000 stored elements in COOrdinate format>
{% endhighlight %}



{% highlight python%}
M = M_sparse.toarray()
M.shape
{% endhighlight %}



{% highlight python%}
    (10000, 10000)
{% endhighlight %}



{% highlight python%}
type(M)
{% endhighlight %}



{% highlight python%}
    numpy.ndarray
{% endhighlight %}


### Spark RDDs and Transformations

The fundamental data structure of Spark is the [resilliant distributed dataset (RDD)](https://spark.apache.org/docs/2.2.0/rdd-programming-guide.html#resilient-distributed-datasets-rdds), which is a fault-tolerant collection of elements that can be operated on in parallel via Spark. The standard method for instantiating an RDD is by referencing a dataset in an external storage system, such as a shared filesystem, HDFS, HBase, or any data source offering a Hadoop InputFormat. Below, we instatiate an RDD using the built-in `textFile` from PySpark. This interprets a text file as a sequence of strings, with each line of the file represented as a single string


{% highlight python%}
conf = SparkConf()
sc = SparkContext(conf=conf)
{% endhighlight %}


{% highlight python%}
lines = sc.textFile('sparse_matrix.txt')
lines.take(10)
{% endhighlight %}



{% highlight python%}
    [u'6.860000000000000000e+02 3.385000000000000000e+03 7.944015774384874939e-01',
     u'5.865000000000000000e+03 5.351000000000000000e+03 7.742883485561377066e-01',
     u'1.590000000000000000e+03 5.723000000000000000e+03 3.410390904855993277e-01',
     u'1.311000000000000000e+03 9.256000000000000000e+03 3.442326085505080790e-01',
     u'9.031000000000000000e+03 4.979000000000000000e+03 9.573724932923319830e-01',
     u'3.627000000000000000e+03 3.573000000000000000e+03 6.118458463822918914e-01',
     u'9.061000000000000000e+03 6.866000000000000000e+03 5.300661428327065883e-01',
     u'1.471000000000000000e+03 7.093000000000000000e+03 8.344234318610610490e-02',
     u'6.158000000000000000e+03 5.673000000000000000e+03 1.340916352995272787e-01',
     u'7.761000000000000000e+03 3.392000000000000000e+03 2.583474112696168001e-01']
{% endhighlight %}


We used the `take(10)` method to view the first 10 items in the RDD, which correspond to the first 10 lines in the file we wrote to disk earlier. We want to convert the lines from strings to 3-tuples. We do this via a transformation on this RDD. The most basic transformation is `map`, which applies a function to every element in the RDD.


{% highlight python%}
M_rdd = lines.map(lambda l: map(float, l.strip().split(' ')))
M_rdd.take(10)
{% endhighlight %}



{% highlight python%}
    [[686.0, 3385.0, 0.7944015774384875],
     [5865.0, 5351.0, 0.7742883485561377],
     [1590.0, 5723.0, 0.3410390904855993],
     [1311.0, 9256.0, 0.3442326085505081],
     [9031.0, 4979.0, 0.957372493292332],
     [3627.0, 3573.0, 0.6118458463822919],
     [9061.0, 6866.0, 0.5300661428327066],
     [1471.0, 7093.0, 0.0834423431861061],
     [6158.0, 5673.0, 0.13409163529952728],
     [7761.0, 3392.0, 0.2583474112696168]]
{% endhighlight %}


So, we successfully created an RDD containing a COO representation of the matrix. 

### Matrix-vector multiplication on Spark RDDs

The basic tranformations on RDDs are `map` and `reduceByKey`, which are exact parallels of the older [MapReduce](https://en.wikipedia.org/wiki/MapReduce) paradigm. Briefly, a MapReduce operation does the following:

1. _Map:_ Apply a function to each element of the input dataset, resulting in a sequence of key-value pairs: $[(k_1, v_1), (k_2, v_2), (k_1, v_3), \ldots]$
2. _Group:_ The key-value pairs are sorted and organized by key, so that each unique key is associated with a list of values: $[(k_1, [v_1, v_3, \ldots]), (k_2, [v_2, \ldots]), \ldots]$
3. _Reduce:_ Combine the values in each key's list according to some function. Function is defined on two values at a time and must be associative and communitive.

For example, the following would be the reduce function used to take the sum over all elements associated with a key:

{% highlight python%}
def summation(v1, v2):
    return v1 + v2
{% endhighlight %}

which can be written more compactly using `lambda` form:

{% highlight python%}
lambda v1, v2: v1 + v2
{% endhighlight %}

As it turns out, the MapReduce paradigm is particularly well-suited to multiplying a sparse matrix and a vector. Let's explore why that is, and then go through an example.

Given the matrix equation

$$y=Ax$$

with $A\in\mathbf{R}^{m\times n}$, each element of $y$ is defined as

$$y_i = \sum_{j=1}^n A_{ij} x_j.$$

So, if we have an RDD representing the matrix, and the vector $x$ fits in memory, then we carry out the multiplication as follows:

1. _Map:_ Take in tuples `(i, j, Aij)` and return tuples `(i, Aij * x[j])`
2. _Group:_ Group all entries by row index
3. _Reduce:_ Sum values for each row index

Spark's `reduceByKey` performs steps 2 and 3 together. All that's left is to correctly organize the results. We must sort the results by key and then handle missing keys, which would occur if a row of our matrix does not contain any non-zero entries. Let's try it out.

First, we create a random vector to multiply against our matrix.


{% highlight python%}
v_in = np.random.random(size=n)
{% endhighlight %}

Next, we perform the MapReduce operation, using Spark. Note how transformations can be chained together. This is not necessary, but is often a cleaner way to represent a multi-step operation. In the last step, we use `collect` which converts the resulting RDD to a Python list. This should be done with care! If the resulting list is too large, this could cause some real problem. In this case, we know the resulting vector is the same size as the input vector, so we can safely collect the RDD to active memory.


{% highlight python%}
v_out_spark_raw = np.array(
    M_rdd\
        .map(lambda x: (x[0], v_in[int(x[1])] * x[2]))\
        .reduceByKey(lambda v1, v2: v1 + v2)\
        .sortByKey()\
        .collect()
)
{% endhighlight %}


{% highlight python%}
len(v_out_spark_raw)
{% endhighlight %}



{% highlight python%}
    8620
{% endhighlight %}


Uh-oh, we we expecting a vector in $\mathbf{R}^{\text{10,000}}$! As mentioned above, this happens when the sparse matrix has no non-zero entries in some rows. We can easily handle this case by using some NumPy indexing tricks, as follows:


{% highlight python%}
v_out_spark = np.zeros(n)
v_out_spark[map(int, v_out_spark_raw.T[0])] = v_out_spark_raw.T[1]
{% endhighlight %}

Finally, we will compare what we just calculated to what we get with Numpy, using the dense array from earlier.


{% highlight python%}
v_out_numpy = M.dot(v_in)
{% endhighlight %}


{% highlight python%}
np.allclose(v_out_spark, v_out_numpy)
{% endhighlight %}



{% highlight python%}
    True
{% endhighlight %}



{% highlight python%}
v_out_numpy[:20]
{% endhighlight %}



{% highlight python%}
    array([0.20550791, 0.24228745, 1.88363129, 0.66752008, 0.01382379,
           0.28009837, 0.52376888, 0.10529744, 0.        , 0.62103075,
           1.07149336, 0.06488723, 0.        , 1.02896754, 0.63032014,
           0.30943638, 0.41731815, 1.30066203, 0.29911015, 0.01944877])
{% endhighlight %}



{% highlight python%}
v_out_spark[:20]
{% endhighlight %}



{% highlight python%}
    array([0.20550791, 0.24228745, 1.88363129, 0.66752008, 0.01382379,
           0.28009837, 0.52376888, 0.10529744, 0.        , 0.62103075,
           1.07149336, 0.06488723, 0.        , 1.02896754, 0.63032014,
           0.30943638, 0.41731815, 1.30066203, 0.29911015, 0.01944877])
{% endhighlight %}


We have a match!
