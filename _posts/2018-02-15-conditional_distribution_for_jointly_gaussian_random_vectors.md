---
layout: post
mathjax: true
title: Derivation of conditional distribution for Jointly Gaussian random vectors
tags: [probability, statistics, gaussian, ee278]
---

_This is based on lectures from EE 278 Statistical Signal Processing at Stanford University_

### Background: Jointly Gaussian Random Vectors

Jointly Gaussian random vectors are generalizations of the one-dimensional Gaussian (or normal) distribution to higher dimensions. Specifically, a vector is said to be jointly Gaussian (j-g) if each element of the vector is a linear combination of some number of i.i.d. standard, normal distributions (Gaussians with zero-mean and a variance of one) plus a bias term. In other words, if $X\in\mathbf{R}^n$ is a JG r.v., then

$$
X_i = \mu_i + \sum_{j=1}^m a_{ij}W_j,\quad\quad \forall\;i=1,\ldots,n,
$$

where $W_j\sim\mathcal{N}(0,1)$ is a standard normal random variable for $j=1,\ldots,m$. In fact, one common way to prove that a random vector is JG is to find some matrix $A\in\mathbf{R}^{n\times m}$ and vector $\mu$ that satisfies

$$
X = A W + \mu,
$$

given $W = \left[\begin{matrix} W_1 & W_2 & \cdots & W_m \end{matrix}\right]^T$. Multivariate normal distributions are parameterized by the location $\mu\in\mathbf{R}^n$ and the covariance $\Sigma=AA^T\in\mathbf{R}^{n\times n}$, yielding the following standard notation:

$$X\sim\mathcal{N}(\mu,\Sigma)$$ 

The probability distribution for this random vector is given as

$$
\begin{equation}
f_X(x_1,\ldots,x_n) = (2\pi)^{-n/2}\left\lvert\Sigma\right\rvert^{-1/2}\exp\left(2-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) \right).
\end{equation}
$$

Finally, we say that two vectors $X\in\mathbf{R}^n$ and $Y\in\mathbf{R}^m$ are j-g, if each individual vector is j-g, and the combined vector $\left[\begin{matrix} X & Y \end{matrix}\right]^T\in\mathbf{R}^{(n+m)}$ is also j-g.

### Conditional distributions

In probability theory and statistical estimation, we often encounter problems where we have two j-g random vectors, and we observe a realization of one of the vectors. Based on this observation, we want to know the location and covariance of the remaining vector. This is called finding the "conditional distribution" of the unobserved vector. There is a [famous formula](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions) that states if $X\in\mathbf{R}^n$ and $Y\in\mathbf{R}^m$ are j-g random vectors, with the joint distribution

$$
\left[\begin{matrix} X \\ Y \end{matrix}\right] \sim \mathcal{N}\left(\left[\begin{matrix} \mu_X \\ \mu_Y \end{matrix}\right], \left[\begin{matrix} \Sigma_{X} & \Sigma_{XY}\\ \Sigma_{YX} & \Sigma_{Y} \end{matrix}\right]\right),
$$

then the distribution of $X$ given $Y=y$ is a multivariate Gaussian with location

$$\bar{\mu} = \mu_X + \Sigma_{XY}\Sigma_{Y}^{-1}(y-\mu_Y)$$

and covariance

$$\bar{\Sigma} = \Sigma_{X}-\Sigma_{XY}\Sigma_{Y}^{-1}\Sigma_{YX}.$$

If one were to try to take up the task of deriving these formulas, one might be tempted to plug equation (1) into the [definition of the conditional probability distribution](https://en.wikipedia.org/wiki/Conditional_probability_distribution):

$$
f_{X\mid Y}\left(x \mid Y=y \right) = \frac{f_{X,Y}(x,y)}{f_Y(y)}.
$$

This approach, while valid, is extremely messy and quite difficult. Proving the result using this approach in the (relatively) simple where $X$ and $Y$ are scalars (so the combined vector is in $\mathbf{R}^2$) turns out to be quite hard. Trying to prove the general case this way is an exercise in masochism.

But, it turns out there's a better way! Jointly Gaussian random vectors have the especially nice property that uncorrelated variables are also independent. (Note, this is not true in general. For example, if $X\sim U(-1, 1)$ and $Y=X^2$, then $X$ and $Y$ are uncorrelated but _not_ independent.) We will exploit this fact about JG random variables to prove the formulas given above.

### The Proof

Let $x\in\mathbf{R}^n$ and $y\in\mathbf{R}^m$ be j-g random vectors. (Note, we are dropping the convention of distinguishing between a random vector $X$ and its realization $x$ for ease of notation.) The joint distribution is given as:

$$
\left[\begin{matrix} x \\ y \end{matrix}\right] \sim \mathcal{N}\left(\left[\begin{matrix} \mu_x \\ \mu_y \end{matrix}\right], \left[\begin{matrix} \Sigma_{x} & \Sigma_{xy}\\ \Sigma_{yx} & \Sigma_{y} \end{matrix}\right]\right)
$$

Let $\tilde{x}=x-\mu_x$ and $\tilde{y}=y-\mu_y$ be the mean-centered versions of $x$ and $y$. Next, introduce $z\triangleq \tilde{x}-A\tilde{y}$. Note that $\mathsf{E}[z]=0$ by construction because $\tilde{x}$ and $\tilde{y}$ are both zero-mean. We can then choose $A$ such that $z$ and $\tilde{y}$ are uncorrelated. Because $z$ and $\tilde{y}$ are also j-g, being uncorrelated implies that they are independent. We find $A$ by setting $\mathsf{Cov}(z,\tilde{y})=\mathsf{E}\left[z\tilde{y}\right] =0$ and solving for $A$.

$$
\begin{align*}
\mathsf{E}\left[z\tilde{y}^T\right] =0 &= \mathsf{E}\left[(\tilde{x}-A\tilde{y})\tilde{y}^T \right] \\
&= \mathsf{E}\left[\tilde{x}\tilde{y}^T\right]-A\cdot\mathsf{E}\left[\tilde{y}\tilde{y}^T\right] \\
&= \Sigma_{xy} - A \Sigma_{y}\quad\implies\quad A = \Sigma_{xy}\Sigma_{y}^{-1}
\end{align*}
$$

By the way we defined $z$, we have $\tilde{x}=A\tilde{y}+z$, with $\tilde{y}$ and $z$ independent. If we condition on $\tilde{y}$, then $\tilde{y}$ is fixed and is no longer random, and $z$ is unaffected, so we have

$$
\begin{align*}
\mathsf{E}[\tilde{x}\mid \tilde{y}] &= A\tilde{y}+\mathsf{E}[z] \\
&=  \Sigma_{xy}\Sigma_{y}^{-1}\tilde{y} + 0\\
&= \Sigma_{xy}\Sigma_{y}^{-1}\tilde{y} \\
\mathsf{E}[x\mid \tilde{y}] &= \mathsf{E}[\tilde{x}\mid \tilde{y}] + \mu_x \\
\implies\quad \mathsf{E}[x\mid y] &= \mu_x + \Sigma_{xy}\Sigma_{y}^{-1}(y-\mu_y) \\
\mathsf{Cov}(x\mid y) &= \mathsf{Cov}(\tilde{x}\mid \tilde{y}) \\
&= \mathsf{Cov}(z) \\
&= \mathsf{E}\left[zz^T\right] \\
&= \mathsf{E}\left[\left(\tilde{x}-A\tilde{y}\right)\left(\tilde{x}^T-\tilde{y}^TA^T\right)\right] \\
&= \mathsf{E}\left[\tilde{x}\tilde{x}^T + \tilde{x}\tilde{y}^TA^T -A\tilde{y}\tilde{x}^T +A\tilde{y}\tilde{y}^TA^T\right] \\
&= \Sigma_{x}-\Sigma_{xy}A^T -A\Sigma_{yx} +A\Sigma_{y}A^T \\
&= \Sigma_{x}-\Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{xy}^T -\Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{yx} +\Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{y}\Sigma_{y}^{-1}\Sigma_{xy}^T \\
&= \Sigma_{x}-\Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{xy}^T
\end{align*}
$$

And so, we have derived the conditional distribution of $x\mid y$:

$$
\begin{align*}
x\mid y \sim \mathcal{N}\left( \mu_x + \Sigma_{xy}\Sigma_{y}^{-1}(y-\mu_y),  \Sigma_{x}-\Sigma_{xy}\Sigma_{y}^{-1}\Sigma_{xy}^T\right)
\end{align*}
$$

Intuitively, when we get information about $y$, we update the expected value and shrink the covariance matrix of $x$. Note that if the covariance between $x$ and $y$ were zero (i.e. the two vectors are independent), then these expressions reduce to the mean and covariance matrix of $x$. This makes sense! If $x$ and $y$ are independent, knowing something about $y$ gives us no information about $x$.