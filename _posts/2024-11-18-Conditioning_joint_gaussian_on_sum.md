---
layout: post
mathjax: true
title: Distribution of a Jointly Gaussian random vector conditioned on observing the sum of entries
tags: [probability, statistics, gaussian, ee278]
---

_This is a follow-up to the [previous post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}) on conditioning a jointly Gaussian probability distribution on partial observations_
### Setting

Let $X$ be a jointly Gaussian (j-g) random vector with mean $\mu\in\mathbf{R}^n$ and covariance matrix $\Sigma\in\mathbf{R}^{n\times n}$, such that $X\sim\mathcal{N}(\mu, \Sigma)$, and let $S$ be the sum of the entries of $X$. We are interested in the setting where we wish to update our belief about the distribuition of $X$ given our observation that the sum equals a specific value, $S=s$.

Because $S$ is a linear transform of j-g randon variables, $S$ is also itself Gaussian. As in the [prior post]({{ "/conditional_distribution_for_jointly_gaussian_random_vectors/" | absolute_url}}), we will exploit the unique property of Gaussians that uncorrelated variables are also independent.

There is a Marimo notebook ([app](https://marimo.io/p/@gismo/notebook-mf27rb), [editable code](https://marimo.app/#code/JYWwDg9gTgLgBCAhlUEBQaD6mDmBTAOzykRjwBNMB3YGACzgF44AiABgDoBODgRi4C05PADcAzCzSIwYJgmSoOAQRkAKAJQY0AAWlgOAYzwAbY2mEAzONlUgI6gFxo4LuFACuBTACN3MGBAEcnYc7sAcvv6BqsaI3iaMLERUcOSkiCyargjumADOxsDCUMEQoeEFRcSqAgBMADRwDXCcvI2x8caJIO4sjSKIxu54jGxZrnn5hcWl5RyVxTXNza3tcQkseX1wA0MjY86uIXR5MIgGANaqANoeXpEBBI09U1VQjZMLxAC64y5QeBg7igQReX3ebk8Pj8jw+r2KWl0MkMJjMlmsmFUBDAjTu0KiBEch3+UIegWJcAAxHB8EQSGQ4Ig4AAFADKABE4AYIAMUIgCEZ5DAUAAPClIYXAEWs4AALzwcl4bApAFk5NiOCQCOQICBNfzyLZSKKZfLnsapaa8H84AAhdX6HUwVQqxoqjjC-l5SB5PAadRwADULT4cAAVHANXgAJ5+iUmuXWinUgyDAzuWIMkQUzAQB0cQJ4PJGyXSxM2kRyVT27TWexwAD0cFUuY4ABU4LWa3WbSm0xnSAqlBSlPmY3GLWX5QGBJH9BA-NURI0NYXi-HLeWbQCgSC4EpGra3ebS1b+oi9CjTOY8FYbEe4GBjDAPgQ8kTsnk3xw6HhSEgwGrRoDAAxIn0QPIkEyCknxgD1aGMP0WGgYAcGAAhBi5HkFH5QUNxFaDslgjgcAMCwNApHdgQIC9kSMa90RsA87UaWDX3fJxsltTBkjkUcu07fd2wpL88h-P8YAA6seLwKhgNAlhwMgjIbWImAEKQ9wwDSMhyCw3lgFwhV8MI1xiNI8jt0Bajm245J6k0DAkX0ei0VvDEgLtGS5LnVjnw-Mzn3mMgQCxR1DJwat1DWTpEhQtCMOMUyXGI048FCjVyAi6Tkmi+QoAuYgLBAGBEgImKNi0nSKGSx8gvUmBENUFgssQHBAkGPI4AgKxuREQy+QFYyLSMLZVKCxDaUNca4IsijsiokFaJc1EbzvTFmPws8EAgZ5cnBFccTgT5pmIfoApcal0O9PADHgKrBz0kA-wIDg4FoY66AXYw9LydwQDgAJuu8X0oBEChjvFXJeOYE63g4XZhnDHYg33QTbH206oARwYkYjVciHXScrXUG0QhEU5ziua4NT+0KXlyvbvN+SjrL3DHvIc5arzc9awr8l8dguqkuV-S5AboUgUeALqmTyUAnzwEVGnCPA3oIQIBCIHBSGAcHGSCOmuqB3gYKCtLQpEGb4Mav0LBYZdjv+xgAG9af+1QrYcDhagsABfWrzLI+bXEWminMvVy1o85i2OOt9hdE8T-2kVRmJA6QwNiZTA6CuarN3cOdEj1bGI2w6BeFvBUN2Rpq5wcGDHzQpEpwDh67oNPrYt1R692a389Zwu4D7wY65r27uajsuNE41wFegeAkBQOxGS6uwKQX2BI3+sBozXudN-ARfjr-bxoCCCD47yI-IG3iUnwgRrgG8Dg98f+Ar9goebLsCu6sFqJLQwB1oYWetgJgzAWDYCQOhbALA54uEvHcCiQA)) accompanying this post that allows you to play with some relevent numerical experiments. It runs in your browser, so feel free to edit and play around!

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

$A$ has rank $n-1$ with exactly one "near-zero" eigenvalue, and so the covariance is always shrunk in a way that depends on the original covariance matrix. This has the effect in which blocks of more highly correlated variable will have their variance (diagonal entries) shrunk more than independent variables. 