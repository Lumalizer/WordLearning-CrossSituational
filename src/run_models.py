from BilingualDataset import BilingualDataset
from CrossSituationalModel import CrossSituationalModel


def run_models(interval_method):
    dataset = BilingualDataset(percentage=100)
    model_p = CrossSituationalModel("Progressive", dataset, interval_method)
    model_p.train()

    dataset.shuffle_dataset()

    model_r = CrossSituationalModel("Random", dataset, interval_method)
    model_r.train()

    # model_p.plot_results(model_r)
    # model_p.plot_results(model_r, difference=True)

    return model_p, model_r