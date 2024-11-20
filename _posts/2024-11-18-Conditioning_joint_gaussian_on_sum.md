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

There is a Marimo notebook ([app](https://marimo.io/p/@gismo/notebook-mf27rb), [editable code](https://marimo.app/#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-s5KJOB2AmDMBYNgJAIlsAsDE2heEKkgA)) accompanying this post that allows you to play with some relevent numerical experiments. It runs in your browser, so feel free to edit and play around!

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