from BilingualDataset import BilingualDataset
from CrossSituationalModel import CrossSituationalModel
from plot_all_results import plot_all_results


def run_models(interval_method):
    dataset = BilingualDataset(percentage=1)
    model_p = CrossSituationalModel("Progressive", dataset, interval_method)
    model_p.train()

    dataset.shuffle_dataset()

    model_r = CrossSituationalModel("Random", dataset, interval_method)
    model_r.train()

    # model_p.plot_results(model_r)
    # model_p.plot_results(model_r, difference=True)

    return model_p, model_r

plot_all_results(run_models("uniquewords"), run_models("totalwords"))