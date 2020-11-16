# An Examination of the State-of-the-Art for Multivariate Time Series Classification
Code repository for workshop paper in [LITSA](https://lipn.github.io/LITSA2020/program), [ICDM 2020](http://icdm2020.bigke.org/).


### Abstract

The UEA Multivariate Time Series Classification (MTSC) archive released in 2018 provides an opportunity to evaluate many existing time series classifiers on the MTSC task. Nevertheless, although many new TSC approaches were proposed recently, a comprehensive overview and empirical evaluation of techniques for the MTSC task is currently missing from the time series literature. In this work, we investigate the state-of-the-art for multivariate time series classification using the UEA MTSC benchmark. We compare recent methods originally developed for univariate TSC, to bespoke methods developed for MTSC, ranging from the classic DTW baseline to very recent linear classifiers (e.g., MrSEQL, ROCKET) and deep learning methods (e.g., MLSTM-FCN, TapNet). We aim to understand whether there is any benefit in learning complex dependencies across different time series dimensions versus treating dimensions as independent time series, and we analyse the predictive accuracy, as well as the efficiency of these methods. In addition, we propose a simple statistics-based time series classifier as an alternative to the DTW baseline. We show that our simple classifier is as accurate as DTW, but is an order of magnitude faster. We also find that recent methods that achieve state-of-the-art accuracy for univariate TSC, such as ROCKET, also achieve high accuracy on the MTSC task, but recent deep learning MTSC methods do not perform as well as expected.


### Running the methods

bash method_name.sh


### Dataset

Please download MTSC data from [here](http://www.timeseriesclassification.com/) and unzip files in data directory.

### Full Paper

[Link](https://www.researchgate.net/publication/344501445_An_Examination_of_the_State-of-the-Art_for_Multivariate_Time_Series_Classification)



### Citation

```
@inproceedings{dhariyal20mtsc,
author={B. Dhariyal and T. Le Nguyen and S. Gsponer and G. Ifrim},
title = {An Examination of the State-of-the-Art for Multivariate Time Series Classification},
year = {2020},
booktitle = {Workshop on Large Scale Industrial Time Series Analysis, ICDM 2020}
}
```


