import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from extensions import get_bboxes, convolute, Delta
from utils import customMutation, create_deap_toolbox, cxDummy, Logger, parse_time
from utils import random_sampling, second_sample
import time




class Tree_base_robust(object):

    def __init__(self, forest_type='dt', ntrees=1, goal='min', nproc=None, random_state=None, verbose=True):
        """

        Parameters
        ----------
        forest_type : str
            Type of forest. Options are ``dt`` for decision (regression) trees, ``rf`` for random forest, ``et`` for
            extremely randomized trees, ``gb`` for gradient boosting. Default is ``dt``.
        ntrees : int, str
            Number of trees to use. Use 1 for a single regression tree, or more for a forest. If ``1`` is selected, the
            choice of ``forest_type`` will be discarded and a single regression tree will be used.
        nproc : int
            Number of processors to use. If not specified, all but one available processors will be used. Each processor
            will process a different tree; therefore there is no benefit in using ``nproc`` > ``ntrees``.
        goal : str
            The optimization goal, "min" for minimization and "max" for maximization. This is used only by the methods
            ``recommend`` and ``get_merit``.
        random_state : int, optional
            Fix random seed.
        verbose : bool, optional.
            Whether to print information to screen. If ``False`` only warnings and errors will be displayed.

        Attributes
        ----------
        y_robust : array
            Expectation of the merits under the specified uncertainties, :math:`E[f(x)]`.
        y_robust_std : array
            Uncertainty in the expectation, estimated as standard deviation (:math:`\sigma`) from the variance
            across trees, :math:`\sigma [E[f(X)]]`.
        std_robust : array
            Standard deviation of the merits under the specified uncertainties, :math:`\sigma [f(x)]`.
        std_robust_std : array
            Uncertainty in the standard deviation, estimated as standard deviation (:math:`\sigma`) from the variance
            across trees, :math:`\sigma [\sigma [f(x)]]`.
        forest : object
            ``sklearn`` object for the chosen ensemble regressor.
        """

        # ---------
        # Init vars
        # ---------
        self.X = None
        self._X = None
        self.y = None
        self._y = None

        self.distributions = None
        self._distributions = None
        self.scales = None
        self.low_bounds = None
        self.high_bounds = None
        self.freeze_loc = None

        self.beta = None
        self._beta = None

        self._ys_robust = None
        self._bounds = None
        self._preds = None

        self._cat_map = None

        self._ys_robust = None
        self._stds_robust = None
        self.y_robust = None
        self.y_robust_std = None
        self.std_robust = None
        self.std_robust_std = None

        self.param_space = None
        self.forest = None

        # ---------------
        # Store arguments
        # ---------------

        # options for the tree
        self.ntrees = ntrees
        self._ntrees = None
        self.max_depth = None
        self.goal = goal
        self.random_state = random_state
        self.forest_type = forest_type

        # other options
        self.nproc = nproc
        if nproc is None:
            self._nproc = cpu_count() - 1  # leave 1 CPU free
        else:
            self._nproc = nproc

        self.verbose = verbose
        if self.verbose is True:
            self.logger = Logger("Golem", 2)
        elif self.verbose is False:
            self.logger = Logger("Golem", 0)
    def fit(self, X, y):
        """Fit the tree-based model to partition the input space.

        Parameters
        ----------
        X : array, list, pd.DataFrame
            Array, list, or DataFrame containing the location of the inputs. It follows the ``sklearn`` format used for
            features: each row :math:`i` is a different sample in :math:`X_{ij}`, and each column :math:`j` is a different
            feature. If the parameters contain categorical variables, please provide a DataFrame.
        y : array, list, pd.DataFrame
            Observed responses for the inputs ``X``.
        """
        self.X = X
        self.y = y
        self._X = self._parse_fit_X(X)
        self._y = self._parse_y(y)

        # determine number of trees and select/initialise model
        self._ntrees = self._parse_ntrees_arg(self.ntrees)
        self._init_forest_model()

        # fit regression tree(s) to the data
        self.forest.fit(self._X, self._y)

        # ----------------------------
        # parse trees to extract tiles
        # ----------------------------
        self._bounds = []
        self._preds = []

        start = time.time()

        if self._nproc > 1:
            # parse all trees, use multiple processors when we have >1 trees
            with ProcessPoolExecutor(max_workers=self._nproc) as executor:
                for _bounds, _preds in executor.map(self._parse_tree, self.forest.estimators_):
                    self._bounds.append(_bounds)
                    self._preds.append(_preds)
        else:
            for i, tree in enumerate(self.forest.estimators_):
                _bounds, _preds = self._parse_tree(tree)
                self._bounds.append(_bounds)
                self._preds.append(_preds)
        end = time.time()
        self.logger.log(f'{self._ntrees} tree(s) parsed in %.2f %s' % parse_time(start, end), 'INFO')