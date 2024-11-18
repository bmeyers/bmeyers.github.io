---
layout: post
mathjax: true
title: Derivation of conditional distribution for Jointly Gaussian random vectors
tags: [probability, statistics, gaussian, ee278]
---

_This is a follow-up to the [previous post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}) on conditioning a jointly Gaussian probability distribution on partial observations_
### Setting

Let $X$ be a jointly Gaussian (j.g.) random vector with mean $\mu\in\mathbf{R}^n$ and covariance matrix $\Sigma\in\mathbf{R}^{n\times n}$, such that $X\sim\mathcal{N}(\mu, \Sigma)$, and let $S$ be the sum of the entries of $X$. Because $S$ is a linear transform of j.g. randon variables, $S$ is also itself Gaussian. We are interested in the setting where we wish to update our belief about the distribuition of $X$ given our observation that $S=s$.

As in the [prior post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}), we will exploit the unique property of Gaussians that uncorrelated variables are also independent.

### The Formula

The distribution of $X$ given the observation of that the sum $S=s$ is given by

$$X\mid S=s \sim \mathcal{N}\left(s\mathbf{1} + A\mu, A\Sigma A^T\right),$$

where $\mathbf{1}\in\mathbf{R}^n$ is the ones vector,

$$v = \frac{1}{\mathbf{1}^T \Sigma \mathbf{1}}\Sigma\mathbf{1},$$

and

$$A=I-v\mathbf{1}^T.$$


### Proof

As asserted above, $(X, S)$ are jointly Gaussian, and so $(AX,S)$ are similarly jointly Gaussian by construction for any matrix $A\in\mathbf{R}^{n\times n}$. We will find matrix $A$ and vector $v\in\mathbf{R}^n$ such that

1) $AX$ is independent from $S$, and
2) $X=AX + Sv$.

If we are able to do this, than the formula above follows from basic definitions. 

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