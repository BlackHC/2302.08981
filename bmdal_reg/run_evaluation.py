import sys
import traceback

from bmdal_reg.evaluation.plotting import *
import typing
from bmdal_reg.evaluation.visualize_lcmd import create_lcmd_plots

# Create a dictionary of LIT_RESULTS_NN_BMDAL with the corresponding labels from LIT_RESULTS_LABELS
# The original results do not contain acquisition batch size ablations for the methods that follow prior art closely.
BEST_RESULTS_NN_BMDAL_DICT = {
    'NN_random': 'Uniform',
    # 'NN_maxdiag_ll_train': r'$\square$ BALD',
    'NN_maxdiag_grad_rp-512_acs-rf-512': r'$\square$ BALD',
    # 'NN_fw-p_ll_acs-rf-hyper-512': r'$\square$ ACS-FW',
    'NN_fw-p_grad_rp-512_acs-rf-hyper-512': r'$\square$ ACS-FW',
    # 'NN_maxdet-p_ll_train': r'$\square$ BatchBALD',
    'NN_maxdet-p_grad_rp-512_train': r'$\square$ BatchBALD',
    # 'NN_bait-fb-p_ll_train': r'$\square$ BAIT',
    'NN_bait-f-p_grad_rp-512_train': r'$\square$ BAIT',
    # 'NN_maxdist-tp_ll': r'$\square$ Core-Set/\\FF-Active',
    'NN_maxdist-p_grad_rp-512_train': r'$\square$ Core-Set/\\FF-Active',
    # 'NN_kmeanspp-p_ll_train': r'$\square$ BADGE',
    'NN_kmeanspp-p_grad_rp-512_acs-rf-512': r'$\square$ BADGE',
    'NN_lcmd-tp_grad_rp-512': r'$\square$ LCMD'
}

LIT_RESULTS_NN_BMDAL_DICT = {
    'NN_random': 'Uniform',
    'NN_maxdiag_ll_train': r'$\square$ BALD',
    'NN_fw-p_ll_acs-rf-hyper-512': r'$\square$ ACS-FW',
    'NN_maxdet-p_ll_train': r'$\square$ BatchBALD',
    'NN_bait-fb-p_ll_train': r'$\square$ BAIT',
    'NN_maxdist-tp_ll': r'$\square$ Core-Set/\\FF-Active',
    'NN_kmeanspp-p_ll_train': r'$\square$ BADGE',
    'NN_lcmd-tp_grad_rp-512': r'$\square$ LCMD'
}

"""
alg_names_relu = ['NN_random', 'NN_maxdiag_grad_rp-512_acs-rf-512', 'NN_maxdet-p_grad_rp-512_train',
                              'NN_bait-f-p_grad_rp-512_train',
                              'NN_fw-p_grad_rp-512_acs-rf-hyper-512', 'NN_maxdist-p_grad_rp-512_train',
                              'NN_kmeanspp-p_grad_rp-512_acs-rf-512',
                              'NN_lcmd-tp_grad_rp-512']
"""

# Create a dictionary of LIT_RESULTS_NN_PREDICTIONS with the corresponding labels from LIT_RESULTS_NN_PREDICTIONS_LABELS
LIT_RESULTS_NN_PREDICTIONS_DICT = {
    'NN_maxdiag_predictions-10': r'$\blacksquare$ BALD',
    'NN_fw-p_predictions-10': r'$\blacksquare$ ACS-FW',
    'NN_maxdet-p_predictions-10': r'$\blacksquare$ BatchBALD',
    'NN_bait-f-p_predictions-10': r'$\blacksquare$ BAIT',
    'NN_maxdist-p_predictions-10': r'$\blacksquare$ Core-Set/\\FF-Active',
    'NN_kmeanspp-p_predictions-10': r'$\blacksquare$ BADGE',
    'NN_lcmd-tp_predictions-10': r'$\blacksquare$ LCMD'
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_VE_CAT_PREDICTIONS_DICT = {
    "VE-CAT_random-20": "Uniform",
    "VE-CAT_maxdiag_predictions-20": r"$\blacksquare$ BALD",
    "VE-CAT_maxdet-p_predictions-20": r"$\blacksquare$ BatchBALD",
    'VE-CAT_bait-f-p_predictions-20': r'$\blacksquare$ BAIT',
    "VE-CAT_fw-p_predictions-20": r"$\blacksquare$ ACS-FW",
    "VE-CAT_maxdist-p_predictions-20": r"$\blacksquare$ Core-Set/\\FF-Active",
    "VE-CAT_kmeanspp-p_predictions-20": r"$\blacksquare$ BADGE",
    "VE-CAT_lcmd-tp_predictions-20": r"$\blacksquare$ LCMD",
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_RF_PREDICTIONS_DICT = {
    "RF_random-100": "Uniform",
    "RF_maxdiag_predictions-100": r"$\blacksquare$ BALD",
    "RF_maxdet-p_predictions-100": r"$\blacksquare$ BatchBALD",
    'RF_bait-f-p_predictions-100': r'$\blacksquare$ BAIT',
    "RF_fw-p_predictions-100": r"$\blacksquare$ ACS-FW",
    "RF_maxdist-p_predictions-100": r"$\blacksquare$ Core-Set/\\FF-Active",
    "RF_kmeanspp-p_predictions-100": r"$\blacksquare$ BADGE",
    "RF_lcmd-tp_predictions-100": r"$\blacksquare$ LCMD",
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_BAGGING_RF_PREDICTIONS_DICT = {
    "BagggingRF_random-10": "Uniform",
    "BagggingRF_maxdiag_predictions-10": r"$\blacksquare$ BALD",
    "BagggingRF_maxdet-p_predictions-10": r"$\blacksquare$ BatchBALD",
    'BagggingRF_bait-f-p_predictions-10': r'$\blacksquare$ BAIT',
    "BagggingRF_fw-p_predictions-10": r"$\blacksquare$ ACS-FW",
    "BagggingRF_maxdist-p_predictions-10": r"$\blacksquare$ Core-Set/\\FF-Active",
    "BagggingRF_kmeanspp-p_predictions-10": r"$\blacksquare$ BADGE",
    "BagggingRF_lcmd-tp_predictions-10": r"$\blacksquare$ LCMD",
}

# Create a dictionary of LIT_RESULTS_VE_CAT_PREDICTIONS with the corresponding labels
LIT_RESULTS_BAGGING_CAT_PREDICTIONS_DICT = {
    "BaggingCAT_random-5": "Uniform",
    "BaggingCAT_maxdiag_predictions-5": r"$\blacksquare$ BALD",
    "BaggingCAT_maxdet-p_predictions-5": r"$\blacksquare$ BatchBALD",
    'BaggingCAT_bait-f-p_predictions-5': r'$\blacksquare$ BAIT',
    "BaggingCAT_fw-p_predictions-5": r"$\blacksquare$ ACS-FW",
    "BaggingCAT_maxdist-p_predictions-5": r"$\blacksquare$ Core-Set/\\FF-Active",
    "BaggingCAT_kmeanspp-p_predictions-5": r"$\blacksquare$ BADGE",
    "BaggingCAT_lcmd-tp_predictions-5": r"$\blacksquare$ LCMD",
}

DEFAULT_FIGSIZE = (6, 6 / 1.62)
WIDE_FIGSIZE = (8, 8 / 1.62)


def plot_all(results: ExperimentResults, alg_names: typing.List[str], with_batch_size_plots: bool = True,
             with_ensemble_size_ablation: bool = True,
             default_ensemble_size: int | None = None,
             with_wb_bb_correlation_plots: bool = False,
             literature_results_dict: typing.Dict[str, str] = LIT_RESULTS_NN_BMDAL_DICT):
    original_results = results
    if default_ensemble_size is not None:
        results = original_results.filter_alg_suffix(f"-{default_ensemble_size}")
    selected_results = results.filter_alg_names(alg_names)
    literature_results = results.filter_alg_names(list(literature_results_dict.keys()))

    print('Generating tables...')
    save_latex_table_all_algs(results, 'table_all_algs.txt')
    save_latex_table_all_algs(selected_results, 'table_selected_algs.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets_lasterror.txt', use_last_error=True)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog.txt', use_log=False)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog_lasterror.txt', use_log=False,
                               use_last_error=True)

    print('Creating learning curve plots...')
    plot_learning_curves_metrics_subplots(results=literature_results, filename='learning_curves_metrics.pdf')
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_rmse_maxe.pdf',
                                  metric_names=['rmse', 'maxe'])
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_q95_q99.pdf',
                                  metric_names=['q95', 'q99'])
    for metric_name in metric_names:
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_{metric_name}.pdf',
                             metric_name=metric_name, figsize=DEFAULT_FIGSIZE)
        plot_learning_curves_ci(results=literature_results, filename=f'learning_curves_literature_{metric_name}_ci.pdf',
                                metric_name=metric_name, figsize=DEFAULT_FIGSIZE)
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_wide_{metric_name}.pdf',
                             metric_name=metric_name, figsize=WIDE_FIGSIZE)

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

    if with_wb_bb_correlation_plots:
        print('Creating correlation plots...')
        for metric_name in metric_names:
            plot_correlation_between_methods_wb_vs_bb(results=selected_results,
                                                      filename=f'last_correlation_between_bb_vs_wb_methods_{metric_name}.pdf',
                                                      metric_name=metric_name, use_last_error=True)
            plot_correlation_between_methods_wb_vs_bb(results=selected_results,
                                                      filename=f'avg_correlation_between_bb_vs_wb_methods_{metric_name}.pdf',
                                                      metric_name=metric_name, use_last_error=False)
    # print('Creating individual learning curve plots...')
    # for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
    #     plot_learning_curves_individual(results=selected_results, metric_name=metric_name)

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
                             metric_name=metric_name, figsize=DEFAULT_FIGSIZE)
            plot_batch_sizes(results=selected_results, filename=f'batch_sizes_wide_{metric_name}.pdf',
                             metric_name=metric_name, figsize=WIDE_FIGSIZE)
        print('Creating individual batch size plots with subplots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual_subplots(results=selected_results,
                                                 filename=f'batch_sizes_individual_{metric_name}.pdf',
                                                 metric_name=metric_name)
        print('Creating individual batch size plots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual(results=selected_results, metric_name=metric_name)

    if with_ensemble_size_ablation:
        # reset the algs
        results = original_results
        alg_names = remove_ensemble_info_from_list(alg_names)
        # print(alg_names)
        filtered_alg_names = list(
            alg_name for alg_name in results.alg_names if any(alg_name.startswith(prefix) for prefix in alg_names))
        selected_results = results.filter_alg_names(filtered_alg_names)
        # literature_results_dict = remove_ensemble_info_from_keys(literature_results_dict)
        # literature_results = results.filter_alg_names(list(alg_name for alg_name in results.alg_names if any(alg_name.startswith(prefix) for prefix in literature_results_dict)))

        # batch size plots
        print('Creating ensemble size plots...')
        # plot_batch_sizes_metrics_subplots(results=selected_results, filename='batch_sizes_metrics.pdf')
        # plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_rmse_maxe.pdf',
        #                           metric_names=['rmse', 'maxe'])
        # plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_q95_q99.pdf',
        #                           metric_names=['q95', 'q99'])
        for metric_name in metric_names:
            plot_ensemble_sizes(results=selected_results, filename=f'ensemble_sizes_{metric_name}.pdf',
                                metric_name=metric_name, figsize=DEFAULT_FIGSIZE)
            plot_ensemble_sizes(results=selected_results, filename=f'ensemble_sizes_wide_{metric_name}.pdf',
                                metric_name=metric_name, figsize=WIDE_FIGSIZE)
        # print('Creating individual batch size plots with subplots...')
        # for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        #     plot_batch_sizes_individual_subplots(results=selected_results,
        #                                          filename=f'batch_sizes_individual_{metric_name}.pdf',
        #                                          metric_name=metric_name)
        # print('Creating individual batch size plots...')
        # for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        #     plot_batch_sizes_individual(results=selected_results, metric_name=metric_name)


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

        if False:
            results.analyze_errors()
            results.analyze_eff_dims()

        if exp_name == 'sklearn':
            try:
                # filter results to only contain algs with RF_ as prefix
                rf_results = results.filter_alg_names(
                    list(filter(lambda alg_name: alg_name.startswith('RF_'), results.alg_names)))
                # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
                alg_names_sklearn = list(LIT_RESULTS_RF_PREDICTIONS_DICT.keys())
                plot_all(
                    rf_results,
                    alg_names=alg_names_sklearn,
                    literature_results_dict=LIT_RESULTS_RF_PREDICTIONS_DICT,
                    with_batch_size_plots=True,
                    with_ensemble_size_ablation=True,
                    default_ensemble_size=100,
                )
            except:
                traceback.print_exc()

            try:
                # filter results to only contain algs with VE-CAT_ as prefix
                rf_results = results.filter_alg_names(
                    list(filter(
                        lambda alg_name: alg_name.startswith('VE-CAT_'), results.alg_names))
                )
                rf_results.exp_name = 'sklearn-ve-cat'
                alg_names_sklearn = list(LIT_RESULTS_VE_CAT_PREDICTIONS_DICT.keys())
                plot_all(
                    rf_results,
                    alg_names=alg_names_sklearn,
                    literature_results_dict=LIT_RESULTS_VE_CAT_PREDICTIONS_DICT,
                    with_batch_size_plots=True,
                    with_ensemble_size_ablation=True,
                    default_ensemble_size=20,
                )
            except:
                traceback.print_exc()

            try:
                # filter results to only contain algs with VE-CAT_ as prefix
                alg_names_list = list(filter(lambda alg_name: alg_name.startswith('BagggingRF_'), results.alg_names))
                rf_results = results.filter_alg_names(alg_names_list)
                rf_results.exp_name = 'sklearn-bagging-rf'
                alg_names_sklearn = list(LIT_RESULTS_BAGGING_RF_PREDICTIONS_DICT.keys())
                plot_all(
                    rf_results,
                    alg_names=alg_names_sklearn,
                    literature_results_dict=LIT_RESULTS_BAGGING_RF_PREDICTIONS_DICT,
                    with_batch_size_plots=True,
                    with_ensemble_size_ablation=True,
                    default_ensemble_size=10,
                )
            except:
                traceback.print_exc()

            # try:
            #     # filter results to only contain algs with VE-CAT_ as prefix
            #     rf_results = results.filter_alg_names(list(filter(lambda alg_name: alg_name.startswith('BaggingCAT_'), results.alg_names)))
            #     rf_results.exp_name = 'sklearn-bagging-cat'
            #     alg_names_sklearn = list(LIT_RESULTS_BAGGING_CAT_PREDICTIONS_DICT.keys())
            #     plot_all(rf_results, alg_names=alg_names_sklearn, literature_results_dict=LIT_RESULTS_BAGGING_CAT_PREDICTIONS_DICT, with_batch_size_plots=False)
            # except:
            #     traceback.print_exc()
        if exp_name == 'relu':
            # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
            alg_names_list = list(BEST_RESULTS_NN_BMDAL_DICT.keys()) + list(
                filter(lambda alg_name: "predictions" in alg_name,
                       results.alg_names))
            relu_results = results.filter_alg_names(alg_names_list)
            alg_names_relu = list(LIT_RESULTS_NN_BMDAL_DICT.keys()) + list(LIT_RESULTS_NN_PREDICTIONS_DICT.keys())
            plot_all(
                relu_results,
                alg_names=alg_names_relu,
                with_batch_size_plots=True,
                with_ensemble_size_ablation=True,
                with_wb_bb_correlation_plots=True,
                literature_results_dict=LIT_RESULTS_NN_BMDAL_DICT | LIT_RESULTS_NN_PREDICTIONS_DICT
            )

            # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
            alg_names_list = list(BEST_RESULTS_NN_BMDAL_DICT.keys()) + list(
                filter(lambda alg_name: "predictions" in alg_name,
                       results.alg_names))

            strongest_relu_results = results.filter_alg_names(alg_names_list)
            strongest_relu_results.exp_name = 'strongest_relu'
            strongest_alg_names_relu = list(BEST_RESULTS_NN_BMDAL_DICT.keys()) + list(LIT_RESULTS_NN_PREDICTIONS_DICT.keys())
            plot_all(
                strongest_relu_results,
                alg_names=strongest_alg_names_relu,
                with_batch_size_plots=True,
                with_ensemble_size_ablation=True,
                with_wb_bb_correlation_plots=True,
                literature_results_dict=BEST_RESULTS_NN_BMDAL_DICT | LIT_RESULTS_NN_PREDICTIONS_DICT
            )

        print('Finished plotting')
        print()

    # print('Creating lcmd visualization...')
    # create_lcmd_plots(n_train=1, n_pool=500, n_steps=20)
