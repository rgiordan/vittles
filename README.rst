#############################################################################
"Variational inference tools to leverage estimand sensitivity": ``vittles``.
#############################################################################

.. image:: https://travis-ci.org/rgiordan/vittles.svg?branch=master
    :target: https://travis-ci.org/rgiordan/vittles

.. image:: https://codecov.io/gh/rgiordan/vittles/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/rgiordan/vittles

Description.
==========================

This is a library (very much still in development) intended to make sensitivity
analysis easier for optimization problems. The core functionality consists of
tools for "folding" and "flattening" collections of parameters -- i.e., for
converting data structures of constrained parameters to and from vectors
of unconstrained parameters.

For background and motivation, see the following papers:

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


Using the package.
==========================

We welcome new users!  However, please be aware that the package is still in
development.  We encourage users to contact the author (github user
``rgiordan``) for advice, bugs, or if you're using the package for something
important.


Installation.
-------------------------

To install the latest tagged version, install with ``pip``:

``python3 -m pip install vittles``.

Note that ``vittles`` is under rapid development, so you may want to
clone the respository and use the master branch instead.


Documentation and Examples.
----------------------------------

.. _readthedocs: https://vittles.readthedocs.io/

For examples and API documentation, see readthedocs_.

Alternatively, check out the repo and run ``make html`` in ``docs/``.
