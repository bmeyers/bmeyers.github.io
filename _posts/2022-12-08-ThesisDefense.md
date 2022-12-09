---
layout: post
mathjax: true
title: A Framework for Signal Decomposition with Applications to Solar Energy Generation
tags: [signal processing, convex optimization, optimization, phd progress]
---

_University Ph.D. Dissertation Defense, Department of Electrical Engineering, Advisor: Stephen Boyd_

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        padding: 10px;
        }
</style>

On December 8, 2022, I successfully defended my Ph.D. dissertation titled, "A Framework for Signal Decomposition with Applications to Solar Energy Generation". The slides for my defense are [available here]({{ "assets/Meyers_SignalDecomp_Presentation.pdf" | absolute_url}}). In large part, this is a presentation of the concepts in [this monograph](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html).

Thank you to my defense committee, [Stephen Boyd](https://web.stanford.edu/~boyd/), [Mert Pilanci](https://web.stanford.edu/~pilanci/), [Adam Brandt](https://profiles.stanford.edu/adam-brandt), [Juan Rivas-Davila](https://profiles.stanford.edu/juan-rivas-davila), and [Steve Eglash](https://profiles.stanford.edu/stephen-eglash).

## Abstract

We consider the well-studied problem of decomposing a vector time series signal into components with different characteristics, such as smooth, periodic, nonnegative, or sparse. We describe a simple and general framework in which the components are defined by loss functions (which include constraints), and the signal decomposition is carried out by minimizing the sum of losses of the components (subject to the constraints). When each loss function is the negative log-likelihood of a density for the signal component, this framework coincides with maximum a posteriori probability (MAP) estimation; but it also includes many other interesting cases. Summarizing and clarifying prior results, we give three distributed optimization methods for computing the decomposition.
 
The signal decomposition (SD) framework has applications across many fields, but we have been motivated by problems related to large-scale data analysis for the photovoltaic (PV) power generation industry.  We will demonstrate a typical example of loss-factor analysis for PV systems using the SD framework. In addition, we will discuss software implementations of both the SD modeling framework and the PV data analysis applications, both of which are published as open-source Python packages.

![png]({{ "assets/BennetAndStephen.jpg" | absolute_url}})