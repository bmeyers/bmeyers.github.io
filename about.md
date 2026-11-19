---
layout: page
title: About
permalink: /about/
---

I'm Bennet Meyers, a Senior Researcher at the [National Laboratory of the Rockies](https://www.nlr.gov) (formerly NREL) and an Adjunct Professor in [Electrical Engineering](https://ee.stanford.edu) at [Stanford University](https://www.stanford.edu).

I use mathematical optimization for signal processing, probabilistic forecasting, and control in electric power systems. I develop methods that can scale from a single site to a regional power grid while keeping estimates and forecasts traceable to the measurements and assumptions behind them.

I earned my PhD in Electrical Engineering at Stanford in 2023. My advisor, [Stephen Boyd](https://web.stanford.edu/~boyd/), and I wrote [Signal Decomposition Using Masked Proximal Operators](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html), a book about separating a measured signal into components such as trends and periodic patterns ([publisher's page](https://www.nowpublishers.com/article/Details/SIG-122)). I [defended my dissertation]({{ "/ThesisDefense/" | absolute_url}}) in December 2022.

From 2018 to 2026, I led [PVInsight](https://bmeyers.github.io/PVInsight/), a U.S. Department of Energy project to help solar operators understand how their systems perform. We developed [Solar Data Tools](https://solar-data-tools.readthedocs.io/), an open-source Python library for analyzing solar power measurements. Our [software paper](https://doi.org/10.21105/joss.08478) describes the library, and our [Phase 2 final technical report](https://doi.org/10.66816/tp4471481) summarizes what we learned.

I continue this work through the [PV Fleet Performance Data Initiative](https://www.nlr.gov/pv/fleet-performance-data-initiative), a program of DOE's [Integrated Energy Systems Office](https://www.energy.gov/cmei/systems/integrated-energy-systems-office). I develop and support signal decomposition methods in two parts of the fleet analysis pipeline. Solar Data Tools works from measured power output alone. In [RdTools](https://github.com/NatLabRockies/rdtools/pull/518), I start with daily energy production that has been normalized using engineering models and weather data. Decomposing that series helps us estimate long-term degradation and losses from soiling.

Since 2024, I've led REGROW (resilient energy generator risk under outlier weather) for the [Office of Electricity Grid Controls Division](https://www.energy.gov/oe/our-organization). We forecast how much renewable power will be available during extreme weather and quantify the uncertainty in those forecasts. A central question is how best to charge and discharge batteries when a rare weather event could limit generation for days. We're developing [CVXOPF](https://github.com/cvxgrp/cvxopf) to study how transmission constraints affect those decisions across the grid.

### Other selected work

- [PV Fleet Modeling via Smooth Periodic Gaussian Copula](https://stanford.edu/~boyd/papers/pv_flt_mod_spgc.html). Mehmet G. Ogut, Bennet Meyers, and Stephen P. Boyd. IEEE Photovoltaic Specialists Conference, 2023. We modeled how power output at different sites varies together over time and used the model to forecast generation, fill gaps in measurements, and detect anomalies.
- [Interpretable Net Load Forecasting Using Smooth Multiperiodic Features](https://web.stanford.edu/~boyd/papers/multiperiodic_forecasting.html). Giray Ogut, Bennet Meyers, and Stephen Boyd. April 2024 manuscript. Our day-ahead model captures how daily net-load patterns vary by weekday and season. It estimates quantiles for individual hours and generates plausible load trajectories for a full day. The [notebook and data](https://github.com/cvxgrp/smooth_multiperiodic_forecasting_experiments) reproduce our experiments on hourly net load in Rhode Island.
- [TSGAM](https://github.com/cvxgrp/tsgam) extends our work on signal decomposition to time-series generalized additive models. With Nimish Telang and David Chassin, I use these models to separate temperature effects from daily and seasonal patterns in electricity demand and to forecast load from those components.
- [cvx-sd-skill](https://github.com/cvxgrp/cvx-sd-skill) includes the `signaldecomp` Python library for building convex signal decomposition models in CVXPY, along with a coding-agent skill and worked examples for fitting and checking those models.
- [Convex Optimization with Smart Grid Examples](https://www.osti.gov/biblio/3018252). Stephen Boyd and Bennet Meyers. In this 2025 IEEE conference talk, we show how to formulate and solve four power-system problems in CVXPY.

You can find more of my papers on [Google Scholar](https://scholar.google.com/citations?hl=en&user=zyJpVssAAAAJ&view_op=list_works&sortby=pubdate) and my software on [GitHub](https://github.com/bmeyers/).

### Contact

You can reach me by email at `bennet.meyersim [at] nlr [dot] gov` or `bennetm [at] stanford [dot] edu`. If I miss your message, feel free to follow up. I rarely respond to cold messages on Twitter or LinkedIn.
