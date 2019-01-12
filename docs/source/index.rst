Documentation for ``vittles``
===============================

Parameter folding and flattening: parameter origami, or ``vittles``.

This is a library (very much still in development) intended to make sensitivity
analysis easier for optimization problems. The core functionality consists of
tools for "folding" and "flattening" collections of parameters -- i.e., for
converting data structures of constrained parameters to and from vectors
of unconstrained parameters.

The purpose is to automate much of the boilerplate required to perform
optimization and sensitivity analysis for statistical problems that employ
optimization or estimating equations.

The functionality of ``vittles`` can be divided into three mutually
supportive pieces:

* Tools for converting structured parameters to and from "flattened"
  representations,
* Tools for wrapping functions to accept flattened parameters as arguments, and
* Tools for using functions that accept flattened parameters to perform
  sensitivity analysis.

A good place to get started is the :ref:`examples`.

For additional background and motivation, see the following papers:

| Covariances, Robustness, and Variational Bayes
| Ryan Giordano, Tamara Broderick, Michael I. Jordan
| https://arxiv.org/abs/1709.02536

|

| A Swiss Army Infinitesimal Jackknife
| Ryan Giordano, Will Stephenson, Runjing Liu, Michael I. Jordan, Tamara Broderick
| https://arxiv.org/abs/1806.00550

|

| Evaluating Sensitivity to the Stick Breaking Prior in Bayesian Nonparametrics
| Runjing Liu, Ryan Giordano, Michael I. Jordan, Tamara Broderick
| https://arxiv.org/abs/1810.06587

.. toctree::
   :maxdepth: 1

   installation
   example_notebooks/examples
   api/api
   release-history
