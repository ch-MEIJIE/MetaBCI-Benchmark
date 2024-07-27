from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.metabci_utils import gen_ssvep_filterbank
    # import your reusable functions here
    import metabci.brainda.algorithms.decomposition as decomposition


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'MetaBCI-docomposition-algo'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "model": [
            "TRCA",
            "FBTRCA",
            "ECCA",
        ],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []
    
    sampling_strategy = "run_once"

    def set_objective(self, X, y, srate, special_data=None):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        self.special_data = special_data
        if self.model[:2] == "FB":
            filterbank, filterweights = gen_ssvep_filterbank(srate=srate)
            self.clf = getattr(decomposition, self.model)(
                filterbank=filterbank, filterweights=filterweights)
        else:
            self.clf = getattr(decomposition, self.model)()

    def run(self, _):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        
        self.clf.fit(self.X, self.y, self.special_data['Yf'])

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.clf)
