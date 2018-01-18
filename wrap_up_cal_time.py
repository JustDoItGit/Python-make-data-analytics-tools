# coding:utf-8
from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats

import timeit


def fill_fre_top_5(x):
    if len(x) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array


def eda_analysis_cal_time(missSet=[np.nan, 9999999999, -999999], df=None):
    # (11)Count distinct#
    start = timeit.default_timer()
    count_un = df.apply(lambda x: len(x.unique()))
    count_un = count_un.to_frame('count')
    print('Count Running Time: %fs' % (timeit.default_timer() - start))

    # (2)Zero Values#
    start = timeit.default_timer()
    count_zero = df.apply(lambda x: np.sum(x == 0))
    count_zero = count_zero.to_frame('count_zero')
    print('Count Zero Running Time: %fs' % (timeit.default_timer() - start))

    # (3)Mean Values#
    start = timeit.default_timer()
    df_mean = df.apply(lambda x: np.mean(x[~np.isin(x, missSet)]))
    df_mean = df_mean.to_frame('mean')
    print('Mean Running Time: %fs' % (timeit.default_timer() - start))

    # (4)Median Values 中位数#
    start = timeit.default_timer()
    df_median = df.apply(lambda x: np.median(x[~np.isin(x, missSet)]))
    df_median = df_median.to_frame('median')
    print('Median Running Time: %fs' % (timeit.default_timer() - start))

    # (5)Mode Values众数#
    start = timeit.default_timer()
    df_mode = df.apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[0][0])
    df_mode = df_mode.to_frame('mode')
    print('Mode Running Time: %fs' % (timeit.default_timer() - start))

    # (6)Mode Percentage#
    start = timeit.default_timer()
    df_mode_count = df.apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[1][0])
    df_mode_count = df_mode_count.to_frame('mode_count')

    df_mode_perct = df_mode_count / df.shape[0]
    df_mode_perct.columns = ['mode_perct']
    print('Mode Percentage Running Time: %fs' % (timeit.default_timer() - start))

    # (7)Min Values#
    start = timeit.default_timer()
    df_min = df.apply(lambda x: np.min(x[~np.isin(x, missSet)]))
    df_min = df_min.to_frame('min')
    print('Min Percentage Running Time: %fs' % (timeit.default_timer() - start))

    # (8)Max Values#
    start = timeit.default_timer()
    df_max = df.apply(lambda x: np.max(x[~np.isin(x, missSet)]))
    df_max = df_max.to_frame('max')
    print('Max Percentage Running Time: %fs' % (timeit.default_timer() - start))

    # (9)quantile values
    start = timeit.default_timer()
    json_quantile = {}

    for i, name in enumerate(df.columns):
        # print('the %d columns: %s' % (i, name))
        json_quantile[name] = np.percentile(df[name][~np.isin(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))

    df_quantife = pd.DataFrame(json_quantile)[df.columns].T
    df_quantife.columns = ['quan01', 'quan05', 'quan25', 'quan50', 'quan75', 'quan95', 'quan99']
    print('quantile Percentage Running Time: %fs' % (timeit.default_timer() - start))

    # (10)Frequent Values
    start = timeit.default_timer()
    json_fre_name = {}
    json_fre_count = {}

    for i, name in enumerate(df.columns):
        index_name = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].index.values
        index_name = fill_fre_top_5(index_name)

        json_fre_name[name] = index_name

        values_count = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].values
        values_count = fill_fre_top_5(values_count)

        json_fre_count[name] = values_count

    df_fre_name = pd.DataFrame(json_fre_name)[df.columns].T
    df_fre_count = pd.DataFrame(json_fre_count)[df.columns].T

    df_fre = pd.concat([df_fre_name, df_fre_count], axis=1)
    df_fre.columns = ['value1', 'value2', 'value3', 'value4', 'value5', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5']
    print('Frequent Percentage Running Time: %fs' % (timeit.default_timer() - start))

    # (11)Miss Values
    start = timeit.default_timer()
    df_miss = df.apply(lambda x: np.sum(np.isin(x, missSet)))
    df_miss = df_miss.to_frame('freq_miss')
    print('Miss Percentage Running Time: %fs' % (timeit.default_timer() - start))

    #####12.Combine All Informations#####
    df_eda_summary = pd.concat(
        [count_un, count_zero, df_mean, df_median, df_mode,
         df_mode_count, df_mode_perct, df_min, df_max, df_fre,
         df_miss, df_quantife], axis=1
    )

    return df_eda_summary
