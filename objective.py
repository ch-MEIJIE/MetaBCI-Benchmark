from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyClassifier
    from metabci.brainda.algorithms.utils.model_selection import (
        EnhancedStratifiedKFold,
    )
    from metabci.brainda.utils.performance import Performance


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "MetaBCI Benchmark"

    # URL of the main repo for this benchmark.
    url = "https://github.com/ch-MEIJIE/MetaBCI-Benchmark"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'random_state': [0],
        'intra_subject': [False],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = ["numpy"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X, y, meta, srate, duration, dataset_name, special_data):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y
        self.meta = meta
        self.srate = srate
        self.duration = duration
        self.dataset_name = dataset_name
        self.special_data = special_data

        self.cv = EnhancedStratifiedKFold(
            n_splits=5,
            shuffle=True,
            return_validate=False,
            random_state=self.random_state,
        )

    def evaluate_result(self, model):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.
        y_pred = model.predict(self.X_test)
        performance = Performance(estimators_list=["Acc", "tITR"], Tw=self.duration)

        result = performance.evaluate(
            y_true=self.y_test, y_pred=y_pred
        )
        accuracy = result["Acc"]
        itr = result["tITR"]

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=accuracy,
            ITR=itr
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        clf = DummyClassifier()
        clf.fit(self.X_train, self.y_train)
        return dict(model=clf)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split(
            self.X, self.y
        )

        return dict(
            X=self.X_train,
            y=self.y_train,
            srate=self.srate,
            special_data=self.special_data,
        )
