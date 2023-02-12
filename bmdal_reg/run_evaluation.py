import sys
import traceback

from bmdal_reg.evaluation.plotting import *
from bmdal_reg.evaluation.visualize_lcmd import create_lcmd_plots

# Create a dictionary of LIT_RESULTS_NN_BMDAL with the corresponding labels from LIT_RESULTS_LABELS
LIT_RESULTS_NN_BMDAL_DICT = {
    'NN_random': 'Uniform',
    'NN_maxdiag_ll_train': r'$\square$ BALD',
    'NN_maxdet-p_ll_train': r'$\square$ BatchBALD',
    'NN_bait-fb-p_ll_train': r'$\square$ BAIT',
    'NN_fw-p_ll_acs-rf-hyper-512': r'$\square$ ACS-FW',
    'NN_maxdist-tp_ll': r'$\square$ Core-Set/\\FF-Active',
    'NN_kmeanspp-p_ll_train': r'$\square$ BADGE',
    'NN_lcmd-tp_grad_rp-512': r'$\square$ LCMD'
}

# Create a dictionary of LIT_RESULTS_NN_PREDICTIONS with the corresponding labels from LIT_RESULTS_NN_PREDICTIONS_LABELS
LIT_RESULTS_NN_PREDICTIONS_DICT = {
    'NN_maxdiag_predictions-10': r'$\blacksquare$ BALD',
    'NN_maxdet-p_predictions-10': r'$\blacksquare$ BatchBALD',
    'NN_bait-f-p_predictions-10': r'$\blacksquare$ BAIT',
    'NN_fw-p_predictions-10': r'$\blacksquare$ ACS-FW',
    'NN_maxdist-p_predictions-10': r'$\blacksquare$ Core-Set/\\FF-Active',
    'NN_kmeanspp-p_predictions-10': r'$\blacksquare$ BADGE',
    'NN_lcmd-tp_predictions-10': r'$\blacksquare$ LCMD'
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_VE_CAT_PREDICTIONS_DICT = {
    "VE-CAT_random": "Uniform",
    "VE-CAT_maxdiag_predictions-1": r"$\blacksquare$ BALD",
    "VE-CAT_maxdet-p_predictions-1": r"$\blacksquare$ BatchBALD",
    'VE-CAT_bait-f-p_predictions-1': r'$\blacksquare$ BAIT',
    "VE-CAT_fw-p_predictions-1": r"$\blacksquare$ ACS-FW",
    "VE-CAT_maxdist-p_predictions-1": r"$\blacksquare$ Core-Set/\\FF-Active",
    "VE-CAT_kmeanspp-p_predictions-1": r"$\blacksquare$ BADGE",
    "VE-CAT_lcmd-tp_predictions-1": r"$\blacksquare$ LCMD",
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_RF_PREDICTIONS_DICT = {
    "RF_random": "Uniform",
    "RF_maxdiag_predictions-1": r"$\blacksquare$ BALD",
    "RF_maxdet-p_predictions-1": r"$\blacksquare$ BatchBALD",
    'RF_bait-f-p_predictions-1': r'$\blacksquare$ BAIT',
    "RF_fw-p_predictions-1": r"$\blacksquare$ ACS-FW",
    "RF_maxdist-p_predictions-1": r"$\blacksquare$ Core-Set/\\FF-Active",
    "RF_kmeanspp-p_predictions-1": r"$\blacksquare$ BADGE",
    "RF_lcmd-tp_predictions-1": r"$\blacksquare$ LCMD",
}

def plot_all(results: ExperimentResults, alg_names: List[str], with_batch_size_plots: bool = True,
             literature_results_dict: Dict[str, str] = LIT_RESULTS_NN_BMDAL_DICT):
    selected_results = results.filter_alg_names(alg_names)
    literature_results = results.filter_alg_names(list(literature_results_dict.keys()))
    literature_names = list(literature_results_dict.values())

    print('Generating tables...')
    save_latex_table_all_algs(results, 'table_all_algs.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets_lasterror.txt', use_last_error=True)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog.txt', use_log=False)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog_lasterror.txt', use_log=False,
                               use_last_error=True)

    print('Creating learning curve plots...')
    plot_learning_curves_metrics_subplots(results=literature_results, filename='learning_curves_metrics.pdf', labels=literature_names)
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_rmse_maxe.pdf',
                                  metric_names=['rmse', 'maxe'])
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_q95_q99.pdf',
                                  metric_names=['q95', 'q99'])
    for metric_name in metric_names:
        plot_learning_curves(results=selected_results, filename=f'learning_curves_{metric_name}.pdf',
                             metric_name=metric_name)
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_{metric_name}.pdf',
                             metric_name=metric_name, labels=literature_names, figsize=(6, 5))
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_wide_{metric_name}.pdf',
                             metric_name=metric_name, labels=literature_names, figsize=(8, 3.5))

    print('Creating individual learning curve plots with subplots...')
    for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        plot_learning_curves_individual_subplots(results=selected_results,
                                                 filename=f'learning_curves_individual_{metric_name}.pdf',
                                                 metric_name=metric_name)
    # TODO error variation plots disabled for now
    if False:
        print('Creating error variation plots...')
        plot_error_variation(results, 'skewness_ri_lcmd-tp_grad_rp-512.pdf', metric_name='rmse',
                             alg_name='NN_lcmd-tp_grad_rp-512',
                             use_relative_improvement=True)

    print('Creating correlation plots...')
    for metric_name in metric_names:
        plot_correlation_between_methods(results=selected_results,
                                         filename=f'correlation_between_methods_{metric_name}.pdf',
                                         metric_name=metric_name)
    print('Creating individual learning curve plots...')
    for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        plot_learning_curves_individual(results=selected_results, metric_name=metric_name)

    if with_batch_size_plots:
        # batch size plots
        print('Creating batch size plots...')
        plot_batch_sizes_metrics_subplots(results=selected_results, filename='batch_sizes_metrics.pdf')
        plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_rmse_maxe.pdf',
                                  metric_names=['rmse', 'maxe'])
        plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_q95_q99.pdf',
                                  metric_names=['q95', 'q99'])
        for metric_name in metric_names:
            plot_batch_sizes(results=selected_results, filename=f'batch_sizes_{metric_name}.pdf',
                             metric_name=metric_name, figsize=(5, 5))
            plot_batch_sizes(results=selected_results, filename=f'batch_sizes_wide_{metric_name}.pdf',
                             metric_name=metric_name, figsize=(6, 3.5))
        print('Creating individual batch size plots with subplots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual_subplots(results=selected_results,
                                                 filename=f'batch_sizes_individual_{metric_name}.pdf',
                                                 metric_name=metric_name)
        print('Creating individual batch size plots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual(results=selected_results, metric_name=metric_name)


if __name__ == '__main__':
    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']
    if len(sys.argv) > 1:
        exp_names = [sys.argv[1]]
    else:
        available_names = utils.getSubfolderNames(custom_paths.get_results_path())
        exp_names = [name for name in available_names if name in ['sklearn', 'relu']]

    for exp_name in exp_names:
        print(f'----- Running evaluation for {exp_name} experiments -----')
        print('Loading experiment results...')
        results = ExperimentResults.load(exp_name)
        print('Loaded experiment results')
        print_avg_results(results)
        # print_all_task_results(results)
        print('Analyzing results')
        results.analyze_errors()
        results.analyze_eff_dims()

        if exp_name == 'sklearn':
            try:
                # filter results to only contain algs with RF_ as prefix
                rf_results = results.filter_alg_names(list(filter(lambda alg_name: alg_name.startswith('RF_'), results.alg_names)))
                # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
                alg_names_sklearn = list(LIT_RESULTS_RF_PREDICTIONS_DICT.keys())
                plot_all(rf_results, alg_names=alg_names_sklearn, literature_results_dict=LIT_RESULTS_RF_PREDICTIONS_DICT, with_batch_size_plots=False)
            except:
                traceback.print_exc()

            try:
                # filter results to only contain algs with VE-CAT_ as prefix
                rf_results = results.filter_alg_names(list(filter(lambda alg_name: alg_name.startswith('VE-CAT_'), results.alg_names)))
                rf_results.exp_name = 'sklearn-ve-cat'
                alg_names_sklearn = list(LIT_RESULTS_VE_CAT_PREDICTIONS_DICT.keys())
                plot_all(rf_results, alg_names=alg_names_sklearn, literature_results_dict=LIT_RESULTS_VE_CAT_PREDICTIONS_DICT, with_batch_size_plots=False)
            except:
                traceback.print_exc()
        if exp_name == 'relu':
            # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
            alg_names_relu = list(LIT_RESULTS_NN_BMDAL_DICT.keys()) + list(LIT_RESULTS_NN_PREDICTIONS_DICT.keys())
            plot_all(results, alg_names=alg_names_relu, with_batch_size_plots=False, literature_results_dict=LIT_RESULTS_NN_BMDAL_DICT | LIT_RESULTS_NN_PREDICTIONS_DICT)


        print('Finished plotting')
        print()

    # print('Creating lcmd visualization...')
    # create_lcmd_plots(n_train=1, n_pool=500, n_steps=20)
