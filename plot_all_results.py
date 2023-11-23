from CrossSituationalModel import CrossSituationalModel
import matplotlib.pyplot as plt
import numpy as np
import time

def plot_all_results(uniquewords: tuple[CrossSituationalModel, CrossSituationalModel], totalwords: tuple[CrossSituationalModel, CrossSituationalModel]):
    u_p, u_r = uniquewords
    t_p, t_r = totalwords

    plt.style.use('ggplot')
    fig, axs = plt.subplots(4, 2, figsize=(10, 14))

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.1, bottom=0.07)

    for ax in axs[:, 0]:
        ax.sharex(axs[0, 0])
        ax.label_outer()

    for ax in axs[:, 1]:
        ax.sharex(axs[0, 1])
        ax.label_outer()

    for ax_row in axs:
        for ax in ax_row:
            ax.sharey(ax_row[0])
            ax.label_outer()

    ax: plt.Axes = axs[0, 0]
    ax.set_ylabel("Words learned")
    ax.set_ylim([0, len(u_p.data.source_unique_words)])
    ax.set_xlim([0, len(u_p.data.source_unique_words)])
    ax.plot(*list(zip(*u_p.results))[:2], label=u_p.name)
    ax.plot(*list(zip(*u_r.results))[:2], label=u_r.name)
    ax.legend()

    ax: plt.Axes = axs[0, 1]
    ax.plot(*list(zip(*t_p.results))[:2], label=t_p.name)
    ax.plot(*list(zip(*t_r.results))[:2], label=t_r.name)
    ax.legend()

    ax: plt.Axes = axs[1, 0]
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    results = list(zip(*u_p.results))
    ax.plot(results[0], results[2], label=u_p.name)
    results = list(zip(*u_r.results))
    ax.plot(results[0], results[2], label=u_r.name)
    ax.legend()

    ax: plt.Axes = axs[1, 1]
    results = list(zip(*t_p.results))
    ax.plot(results[0], results[2], label=t_p.name)
    results = list(zip(*t_r.results))
    ax.plot(results[0], results[2], label=t_r.name)
    ax.legend()

    ax: plt.Axes = axs[2, 0]
    ax.set_ylabel("Words learned (relative)")
    results = list(zip(*u_p.results))
    results2 = list(zip(*u_r.results))
    relative = np.array(results[1]) / np.array(results2[1][:len(results[1])])
    maximum = max((max([i for i in relative.tolist() if i > 0][1:])+ .2),1) or 1
    ax.set_ylim([0, maximum])
    ax.plot(results[0], relative, label="Progressive / Random")
    ax.legend()

    ax: plt.Axes = axs[2, 1]
    results = list(zip(*t_p.results))
    results2 = list(zip(*t_r.results))
    ax.plot(results[0], np.array(results[1]) / np.array(results2[1][:len(results[1])]), label="Progressive / Random")
    ax.legend()

    ax: plt.Axes = axs[3, 0]
    ax.set_xlabel("Unique words processed")
    ax.set_ylabel("Accuracy (relative)")
    results = list(zip(*u_p.results))
    results2 = list(zip(*u_r.results))
    ax.plot(results[0], np.array(results[2]) - np.array(results2[2][:len(results[2])]), label="Progressive - Random")
    ax.legend()

    ax: plt.Axes = axs[3, 1]
    ax.set_xlabel("Total words processed")
    results = list(zip(*t_p.results))
    results2 = list(zip(*t_r.results))
    ax.plot(results[0], np.array(results[2]) - np.array(results2[2][:len(results[2])]), label="Progressive - Random")
    ax.legend()

    plt.savefig(f"results/{time.time()}all.png")
    plt.show()