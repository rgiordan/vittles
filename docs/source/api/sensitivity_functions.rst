
Sensitivity Functions
==========================================

Hyperparameter sensitivity linear approximation
-------------------------------------------------

.. autoclass:: vittles.sensitivity_lib.HyperparameterSensitivityLinearApproximation
  :members:

Hyperparameter sensitivity Taylor series approximation
---------------------------------------------------------

.. autoclass:: vittles.sensitivity_lib.ParametricSensitivityTaylorExpansion
  :members:

Sparse Hessians
----------------------------

.. autoclass:: vittles.sensitivity_lib.SparseBlockHessian
  :members:

Linear response covariances
----------------------------

.. autoclass:: vittles.sensitivity_lib.LinearResponseCovariances
  :members:

DerivativeTerm class
---------------------------

This class is used in the internals of ``ParametricSensitivityTaylorExpansion``.

.. autoclass:: vittles.sensitivity_lib.DerivativeTerm
  :members:
