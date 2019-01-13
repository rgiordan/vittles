Documentation for ``vittles``
===============================

"Variational inference tools to leverage estimator sensitivity" or ``vittles``.

This is a library (very much still in development) intended to make sensitivity
analysis easier for optimization problems.

The purpose is to automate much of the boilerplate required to perform
optimization and sensitivity analysis for statistical problems that employ
optimization or estimating equations.

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
   api/api
   example_notebooks/examples
   release-history
