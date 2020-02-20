---
layout: post
mathjax: true
title: PVInsight
tags: [pvinsight, signal processing, pv data, pv system, data science, funded research]
---

_A Toolkit for Unsupervised PV System Loss Factor Analysis_

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        padding: 10px;
        }
</style>


## Background
Access to increasing volume of photovolatic PV system performance data creates opportunities for monitoring system health and optimizing operations and maintenance (O&M) activities. Analyzing production data from installed PV systems allows for non-intrusive, remote, and automated assessment of performance issues. Doing this at scale will allow us to ensure that the large volume of distributed, rooftop PV systems being installed have good reliability and that they constitute a dependable grid resource, not a destabilizing burden for grid operators. However, standard approaches to analyzing PV performance require:
- A significant amount of engineering time
- Knowledge of PV system modeling science and best practices
- Accurate system configuration information
- Access to reliable irradiance and meteorological data
While these requirements are typically met for large, utility-scale PV systems, distributed, rooftop systems do not generally meet these requirements, and are therefore being mostly ignored by digital O&M companies, to the detriment of these systems and the loss of value to their owners.

## Project Description
We seek to develop algorithms to automate loss factor estimations and performance analysis for small and medium-sized PV systems. Drawing from the disciplines of optimization, signal processing, and machine learning, we are developing novel solutions to difficult data problems and implementing these solutions in an open-source software toolkit, written in Python. These tools will allow users to process data from hundreds of thousands of unique PV systems, automatically detecting operational issues and degradation patterns and forecasting system power production. 

## Software

 - `solar-data-tools` ([link](https://github.com/slacgismo/solar-data-tools)): Tools for performing common tasks on solar PV data signals. These tasks include finding clear days in a data set, common data transforms, and fixing time stamp issues. These tools are designed to be automatic and require little if any input from the user. Libraries are included to help with data IO and plotting as well. 
 - `statistical-clear-sky` ([link](https://github.com/slacgismo/StatisticalClearSky)): Statistical estimation of a clear sky signal from PV system power data.
 - `pv-system-profiler` ([link](https://github.com/slacgismo/pv-system-profiler)): system latitude, longitude, tilt, and azimuth estimation
 - `optimal-signal-demixing` ([link](https://github.com/bmeyers/optimal-signal-demixing)): Modleing Language for finding optimal solutions to structured signal demixing problems

## Papers

- B. Meyers, M. Tabone, and E. C. Kara, "Statistical Clear Sky Fitting Algorithm," in 2018 IEEE 7th World Conf. on Photovol. Energy Conversion, [arXiv:1907.08279](https://arxiv.org/abs/1907.08279)
- B. Meyers, M. Deceglie, C. Deline, and D. Jordan, "Signal processing on PV time-series data: Robust degradation analysis without physical models," _IEEE Journal of Photovoltaics_, 2019. [doi.org/10.1109/JPHOTOV.2019.2957646](https://doi.org/10.1109/JPHOTOV.2019.2957646)

## Funding

This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under Solar Energy Technologies Office (SETO) Agreement Number 34368.