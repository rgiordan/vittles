from vittles.pattern_containers import \
    PatternDict, PatternArray, \
    register_pattern_json, get_pattern_from_json, save_folded, load_folded
from vittles.numeric_array_patterns import NumericArrayPattern
from vittles.psdmatrix_patterns import PSDSymmetricMatrixPattern
from vittles.function_patterns import FlattenedFunction, Functor
from vittles.simplex_patterns import SimplexArrayPattern
from vittles.optimization_lib import \
    PreconditionedFunction, \
    OptimizationObjective
from vittles.sensitivity_lib import \
    HyperparameterSensitivityLinearApproximation, \
    ParametricSensitivityTaylorExpansion, \
    LinearResponseCovariances
import vittles.autograd_supplement_lib

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
