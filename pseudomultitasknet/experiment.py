EXPERIMENT_REGISTRY = {}

def RegisterExperiment(name):
    def decorator(f):
        EXPERIMENT_REGISTRY[name] = f
        return f

    return decorator

class Experiment:
    """Runnable experiment definition

    Each experiment is defined by its name, experiments of the same name
    overwrite eachother.
    """

    def __init__(self):
        self.trainers = []
        self.experiment_functions = []

    def run(self, training=True):
        """Runs one or more experiment functions
        """
        self.register_hooks()

        if training:
            self._train()
        else:
            self._load()

        for func, args, kwargs in self.experiment_functions:
            func(self, *args, **kwargs)

    def _train(self):
        for trainer, dataset, max_epochs in self.trainers:
            trainer.run(dataset, max_epochs=max_epochs)

    def load(self):
        pass

    def register_trainer(self, trainer, dataset, max_epochs):
        self.trainers.append((trainer, dataset, max_epochs))

    def register_experiment_function(self, function, *args, **kwargs):
        self.experiment_functions.append((function, args, kwargs))
