from plot_all_results import plot_all_results
from run_models import run_models

unique = run_models("uniquewords")
total = run_models("totalwords")
plot_all_results(unique, total)