
class EnsembleAgent(object):

    models = {}
    num_models = 0

    def __init__(self, all_models=None):

        if all_models is None:
            raise AttributeError("You should provide at least one model to the ensemble")

        for i in range(len(all_models)):
            self.models[i] = all_models[i]

        self.num_models = len(all_models)

    def predict(self, model_key, input):
        return self.models[model_key].predict(input)

    def predict_all(self, input):
        results = dict()

        for i in len(self.num_models):
            results[i] = self.models[i].predict(input)

        return results
