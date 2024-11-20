---
layout: post
mathjax: true
title: Distribution of a Jointly Gaussian random vector conditioned on observing the sum of entries
tags: [probability, statistics, gaussian, ee278]
---

_This is a follow-up to the [previous post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}) on conditioning a jointly Gaussian probability distribution on partial observations_

[Note: If the math notation is not rendering correctly, try following the steps described [here](https://physics.meta.stackexchange.com/a/14409) to set your MathJax renderer to "Common HTML".]
### Setting

Let $X$ be a jointly Gaussian (j-g) random vector with mean $\mu\in\mathbf{R}^n$ and covariance matrix $\Sigma\in\mathbf{R}^{n\times n}$, such that $X\sim\mathcal{N}(\mu, \Sigma)$, and let $S$ be the sum of the entries of $X$. We are interested in the setting where we wish to update our belief about the distribuition of $X$ given our observation that the sum equals a specific value, $S=s$.

Because $S$ is a linear transform of j-g randon variables, $S$ is also itself Gaussian. As in the [prior post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}), we will exploit the unique property of Gaussians that uncorrelated variables are also independent.

There is a Marimo notebook ([app](https://marimo.io/p/@gismo/notebook-mf27rb), [editable code](https://marimo.app/#code/JYWwDg9gTgLgBCAhlUEBQaD6mDmBTAOzykRjwBNMB3YGACzgF44AiABgDoBODgRi4C05PADcAzCzSIwYJgmSoOAQRkAKADaIAnhACuMTADNg6vIxaad+gM4B6AhDIAjCBADWHHCnIcAVtYgCFgBKDDQAAWkwDgBjPHV1NGFDOGxVEAhggC40ODy4KF0CTCd9GEC5DI5dYA5SmHKCDUQnePMiKjhyUkQQ3PyQXUxrdWBhKEqIatqRseJVAQAmABo4FbhOXlXNVvVzQZZVkUR1XTM2UPy4a2HR8cnpjlnxhfX1ze2WtpZrQ7hj07nS4DKZ0awwRAxNyqADahWK9Uaq0GtzmUFWN2exAAusC8lA8DBdFACAghlj0QUiiUyoEMajxmFIjJYvFEslUphVAQwKt4TSGoFsv18dTEYERXAAMRwfBEEhkOCIOAABQAygAROAxCDHFCIAhxeQwFAAD0lSBNwFNauAAC88HJeGxJQBZOQ8jgkAjkCAgL0G8jpUhm20O5Eh61hvB4uAAIQ90V9MFUrtWro4JoN1kg1jwqmCwTgAGoNnw4AAqOCevBafOW0P2mOSmUxE4xXSaRUiSWYCCJjiBPDWYNWm1N2MiOSqBPhVKZOC2OCqPscAAqcDns-nsdb7c7pEdSklSgHtfrkfHDqLAmr0T0ZCgqhEq09Q5HDajE9jBKJJLgSirHG6YRmO0ZHEyUSsgkSR4CkaTAXAYDqDAGIENYwpXNY6EcHQeCkEgYAzqsMSEeYyGINYSB9FcyEwJmtCmKoLDQMAODAAQJzarqCgGkan6mjR+R0Z4MSGAWkq-sSBCQSycQwRyaSAfGqx0WhGE5FccaYB0cinlum4AeukrYdYuH4TAhEzjpeBUCRZEsBRVG9LGIkwIx+YsLoYDdGQ5DcXqwB8Y6AlCXkIk4GJElXFJ-7WR0yyhBgzLRPJ7JwZyxHxjZdl3qpKGYcJKFPGQIDckmQU4DOwSfLs5isexnHqGFSHFeCeBlZ65CVfFtk1fIUBuMQhggDA5gAMK8IJ2wcXBo0TbwAgCH8OzfN5vkUC1bkecx3WIDggQnNYcAQCkOoiEF+qGiFkZxL8rnFaYcpBg99GReJP6EtJsmpWysHwVyiHaQleWtTAhV5JgAKrFDeAxAOoxNTgHB4GxdDVZKIntWVUMnLV3wNRxJwtbj6gwyIcMI0T6jI6jODo8DfWY21pUrtDA1DVAI1jSwk3TXAiNzTzk1LStXx7F5PmHuQW3FVo1htkxFgQDgsv0e5MBKwAomxhAAmcx2nQFvHXcaKB3WrHBPYQL3M29UWfX+MnJVBaX-ZlykCeBZIMsQr68mD9IUkcEPSnAHE5nD8DrdLCD4QQHDh-A1h0Ho6j+dYuggHA5QnU4eZQBTGcWkMunMJidzEBw+uOlWU6lvpy4ohS1cnGclZ3oORAfpe0aFi2CBTCI4KQtCMKepnZUoh0-XT7ZuJ2yVHXpOSldQK3gId2+3ejo2174xLhNNZb2MrzlB-mDHfl-Egg3DfNvNTSts3cwtouvVbeDPdFRXqzthgsBAPHf4UdoDWAxFnE6KQr4UDjgaOQABvCeWcz4zyyBwRYhgAC+lt3o-3xF9OKc87JJQiK7P6ikuQ8nyqhf4odWx4ShDnOgpB-jh2OsqawoBkJ4FNKsWoeBE4OAIAIIgOBSDAApkqUkk9jq514IvU+IgP4ayYgAl81ws6MCQdESez5sgYOwbgh2klCHOzIXJChGUlI0PUqHUy5kCLSFUMpUi0hyKaGcpbVRnlXSXgAsYj6pinY-WgulAGLj-Y0NDnTdmsTKbME9IjE4tM0YuI-qfWJJwVE7RYDrOUNdDYpCUIE-BBQzFwCyWTSpbEKYxFCW7Shu8vzhjvKHT03pfT+jzBQVQAAWRYP5Ax+mGIgcApgBydL9AGH0TQACsbBQJ72bDFCpqgpkgFGeMvAiUGlWIiZKYChzz4WlIMMAAjrAZYpywJNmuVcYh9yBirzRE8vI1DJLDM2dYMZPDrmhzACgaAmBurgmnJKB5ZzrCXJTHGIsc4NlbJ4cZK4VxSxnxbjXLe94d5e2-DCLI-sOAdEQKaYA1hsSSmCCi4SEBwQgvJfAZgo4LmwF6lQeFBQvlItMOuEsPsOgEqJSSslFLqVrmCdJJCdKDCgtoYC4AwK5V7IUtYqhAc1JISBVABl4IAUs2XsgsqCqlWMtWKS8ljBeD91osVHxzE1SQKNoQK0w4oFwAWdcX5pgilasVRMOVKB6iKqCB-PBjtvou0saqiJ1DpX0rle06ISBTTlSeCg3MsqzVKlFVawscBbyehABxNNejM26toRa6weaI0khVeEzKcbNXlsTZpX+S9Oq6IzTKit5rc3WpyZrTyjrs7OoIK631nqfnbN9Zm4g-quiMqDfoENJMdQ+l4OC1F+Qi2ktLd2hN2aq01oLZ3YtTQjWqBbUe-tNrt1wAADzMF4HgAQfSqW9nXeQRYA4gF7QvdEFoI5L3XvBH2y1A6ixPrLAADjmZKYA8Ev2bsDKkL9iw23bpEloUYZUBBbDgNaxe4bJV1qjb9GNmVQ7cOgPAW+qAlTHQyAh8AtHqxZzAFoRjd4WOQFgNcfCLh-yUWuOhXjbHLTIUcKMJwHBONSfgCJuipHSQZCiYHUT1h63uzSDyUOHJLSspTJaUOVwZTjT9GAFdBAcCLv2odUYdoJESnvZUwpqxRCgKgMdRJ0Rkk0xRmkkzEL8gyi1uhYkHFbPQt0MgR0UBXB0f8bwxlWnXOUQLvAVQog25uoAHzMAuBwE46gylYRhZgASchPMxHKN57FTwYXZcKZypJ1NkYcREM1rzGEQsEKdtcCrAkVNNyhTC3ZGBEOck4kA7ATBmAsGwEgDi2AWCYaVCyeEEkgA)) accompanying this post that allows you to play with some relevent numerical experiments. It runs in your browser, so feel free to edit and play around!

### The Formula

The distribution of $X$ given the observation that the sum $S=s$ is given by

$$X\mid S=s \sim \mathcal{N}\left(sv + A\mu, A\Sigma A^T\right),$$

where,

$$v = \frac{1}{\mathbf{1}^T \Sigma \mathbf{1}}\Sigma\mathbf{1},$$

$$A=I-v\mathbf{1}^T,$$

and $\mathbf{1}\in\mathbf{R}^n$ is the ones vector.

### Proof

As asserted above, $(X, S)$ are j-g, and so $(AX,S)$ are similarly j-g by construction for any matrix $A\in\mathbf{R}^{n\times n}$. We will find matrix $A$ and vector $v\in\mathbf{R}^n$ such that

1. $AX$ is independent from $S$, and
2. $X=AX + Sv$.

If we are able to do this, then the formula above follows from basic definitions. 

$AX$ is independent from $S$ if and only if they are uncorrelated, _i.e._, their covariance matrix is zero:

$$E[A(X-\mu)(S-E[S])] = 0.$$

We know that $S=\mathbf{1}^TX$ and $E[S]=\mathbf{1}^T\mathbf{1}\mu$, so it follows that,

$$E[A(X-\mu)(X-\mu)^T\mathbf{1}] = A\Sigma\mathbf{1}=0.$$

Next, we oberve that

$$
\begin{align*}
X &=AX + Sv \\
AX &=X - Sv \\
&=X-\mathbf{1}^TXv \\
&= (I-v\mathbf{1}^T)X,
\end{align*}
$$

implying that $A=I-v\mathbf{1}^T$, thus proving the second result above. Multipying through by $\Sigma\mathbf{1}$ and noting that $A\Sigma\mathbf{1}=0$, we find

$$
\begin{align*}
0 &= \Sigma\mathbf{1} - v\mathbf{1}^T\Sigma\mathbf{1}  \\
v\mathbf{1}^T\Sigma\mathbf{1} &= \Sigma\mathbf{1} \\
v &= \frac{1}{\mathbf{1}^T \Sigma \mathbf{1}}\Sigma\mathbf{1},
\end{align*}
$$

thus proving the first result. 

### Discussion

Note that both $v$ and $A$ do not depend at all on $s$ and can be pre-calculated. The entries of $v$ are all on the interval $[0,1]$ and, in fact, form a simplex (their values sum to $1$). The posterior mean is always updated to be exactly consistent with the observed sum.

The updated covariance matrix is always "shrunk," _i.e._, $\Sigma - A\Sigma A^T \succeq 0$, so that the uncertainty is reduced in the posterior distribution. Assuming $\Sigma$ is rank $n$, $A$ has rank $n-1$, and the updated covariance becomes degenerate (singular), also with rank $n-1$. This degeneracy is important! It establishes a subspace in $\mathbf{R}^n$ (the nullspace of the updated covariance matrix) along which our posterior distribution has no variance. This subspace is one dimensional with the basis matrix, $\mathbf{1}\in\mathbf{R}^{n\times 1}$. That means the posterior distribution has no variability in the sum of entries. Any sample of this posterior distribution will have the exact same sum!

#### See also

<sub><sup>[https://math.stackexchange.com/a/2942689](https://math.stackexchange.com/a/2942689)</sup></sub>\
<sub><sup>[https://stanford.edu/class/ee363/lectures/estim.pdf](https://stanford.edu/class/ee363/lectures/estim.pdf)</sup></sub>