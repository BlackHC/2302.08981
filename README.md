# Black-Box Batch Active Learning for Regression
https://arxiv.org/abs/2302.08981 https://openreview.net/forum?id=fvEvDlKko6

If you use this code for research purposes, please cite "[Black-Box Batch Active Learning for Regression](https://arxiv.org/abs/2302.08981)":


```bibtex
@misc{kirsch2023blackbox,
    title={Black-Box Batch Active Learning for Regression},
    author={Andreas Kirsch},
    year={2023},
    eprint={2302.08981},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

This repository extends the repository of the paper "[A Framework and Benchmark for Deep Batch Active Learning for Regression](https://arxiv.org/abs/2203.09410)" to supporting empirical covariance kernels which allows to use it for active-learning with black-box models.

See the [GitHub repository](https://github.com/dholzmueller/bmdal_reg) of "A Framework and Benchmark for Deep Batch Active Learning for Regression" for more infos. The following is based on the original readme.

## License

This source code is licensed under the Apache 2.0 license. However, the implementation of the acs-rf-hyper kernel transformation in `bmdal/features.py` is adapted from the source code at [https://github.com/rpinsler/active-bayesian-coresets](https://github.com/rpinsler/active-bayesian-coresets), which comes with its own (non-commercial) license. Please respect this license when using the acs-rf-hyper transformation directly from `bmdal/features.py` or indirectly through the interface provided at `bmdal/algorithms.py`.

### Manually

For certain purposes, especially trying new methods and running the benchmark, it might be helpful or necessary to modify the code. For this, the code can be manually installed via cloning this repository and then following the instructions below:

The following packages (available through `pip`) need to be installed:
- General: `torch`, `numpy`, `dill`
- For running experiments with `run_experiments.py`: `psutil`
- For plotting the experiment results: `matplotlib`, `seaborn`
- For downloading the data sets with `download_data.py`: `pandas`, `openml`, `mat4py`

If you want to install PyTorch with GPU support, please follow the instructions [on the PyTorch website](https://pytorch.org/get-started/locally/). The following command installs the versions of the libraries we used for running the benchmark:
```
pip3 install -r requirements.txt
```
Alternatively, the following command installs current versions of the packages:
```
pip3 install torch numpy dill psutil matplotlib seaborn pandas openml mat4py
```

Clone the repository (or download the files from the repository) and change to its folder:
```
git clone git@github.com:dholzmueller/bmdal_reg.git
cd bmdal_reg
```
Then, copy the file `custom_paths.py.default` to `custom_paths.py` via
```
cp custom_paths.py.default custom_paths.py
```
and, if you want to, adjust the paths in `custom_paths.py` to specify the folders in which you want to save data and results.

## Downloading data

If you want to use the benchmark data sets, you need to download and preprocess them. We do not provide preprocessed versions of the data sets to avoid copyright issues, but you can download and preprocess the data sets using
```
python3 download_data.py
```
Note that this may take a while. This depends of course on your download speed. The preprocessing is mostly fast, but for the (large) methane data set, it took around five minutes and 25 GB of RAM for us. If you cannot download/process the data due to limited RAM, please contact the main developer (see below).

## Usage

Depending on your use case, some of the following introductory Jupyter notebooks may be helpful:
- [examples/benchmark.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/benchmark.ipynb) shows how to download or reproduce our experimental results, how to benchmark other methods, and how to evaluate the results.
- [examples/using_bmdal.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/using_bmdal.ipynb) shows how to apply our BMDAL framework to your use-case.
- [examples/framework_details.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/framework_details.ipynb) explains how our BMDAL framework is implemented, which may be relevant for advanced usage.
- [examples/nn_interface.ipynb](https://github.com/dholzmueller/bmdal_reg/blob/main/examples/nn_interface.ipynb) shows how our NN configuration can be used (without active learning) through a simple scikit-learn style interface.

Besides these notebooks, you can also take a look at the code directly. The more important parts of our code are documented with docstrings.

## Code structure

The code is structured as follows:
- The `bmdal` folder contains the implementation of all BMDAL methods, with its main interface in `bmdal/algorithms.py`.
- The `evaluation` folder contains code for analyzing and plotting generated data, which is called from `run_evaluation.py`.
- The `examples` folder contains Jupyter Notebooks for instructive purposes as mentioned above.
- The file `download_data.py` allows for downloading the data, `run_experiments.py` allows for starting the experiments, `test_single_task.py` allows for testing a configuration on a data set, and `rename_algs.py` contains some functionality for adjusting experiment data in case of mistakes. 
- The file `check_task_learnability.py` has been used to check the reduction in RMSE on different data sets when going from 256 to 4352 random samples. We used this to sort out the data sets where the reduction in RMSE was too small, since these data sets are unlikely to make a substantial difference in the benchmark results.
- The files `data.py`, `layers.py`, `models.py`, `task_execution.py`, `train.py` and `utils.py` implement parts of data loading, training, and parallel execution.

## Original Contributors

- [David Holzmüller](https://www.isa.uni-stuttgart.de/en/institute/team/Holzmueller/) (main developer)
- [Viktor Zaverkin](https://www.itheoc.uni-stuttgart.de/institute/team/Zaverkin/) (contributed to the evaluation code)

If you would like to contribute to the code or would be interested in additional features, please contact David Holzmüller.








