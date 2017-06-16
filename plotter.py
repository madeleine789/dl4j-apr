#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
matplotlib.rc('font', family='Arial')


d2v_results = {}
d2v_results['dutch'] = (13.36, 14.96, 21.15, 12.17, 12.15)
d2v_results['english'] = (16.39, 22.23, 15.70, 15.04, 14.84)
d2v_results['italian'] = (16.36, 17.35, 17.70, 12.47, 16.15)
d2v_results['spanish'] = (17.11, 20.41, 16.29, 18.73, 16.46)

bow_results = {}
bow_results['dutch'] = (26.82, 32.03, 36.60, 35.38, 21.44)
bow_results['english'] = (25.23, 34.53, 28.27, 23.63, 18.46)
bow_results['italian'] = (31.78, 38.04, 28.30, 28.86, 26.22)
bow_results['spanish'] = (18.28, 38.69, 19.68, 18.70, 17.87)

bm_results = {}
bm_results['dutch'] = (18.96, 31.41, 16.78, 15.18, 20.58)
bm_results['english'] = (19.75, 35.03, 23.02, 18.304, 16.87)
bm_results['italian'] = (21.41, 34.22, 21.90, 21.87, 25.61)
bm_results['spanish'] = (19.78, 34.67, 19.05, 22.18, 18.47)

t2v_results = {}
t2v_results['dutch'] = (16.98, 22.26, 15.97, 14.29, 13.22)
t2v_results['english'] = (16.98, 23.23, 16.70, 15.55, 15.98)
t2v_results['italian'] = (18.36, 20.22, 18.56, 13.50, 17.22)
t2v_results['spanish'] = (17.82, 21.48, 16.35, 19.27, 17.07)


def plot_results(results=bm_results, filename="bm", model="bigramowego"):
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, results['dutch'], width, color='#64dd17')
    rects2 = ax.bar(ind + width, results['italian'], width, color='#f50057')
    rects3 = ax.bar(ind + 2*width, results['english'], width, color='#ffc107')
    rects4 = ax.bar(ind + 3*width, results['spanish'], width, color='#2196f3')

    # add some text for labels, title and axes ticks
    ax.set_ylabel(u'RMSE [%]'.encode('utf-8'))
    ax.set_title(u'Wyniki dla modelu ' + model)
    ax.set_xticks(ind + 2 * width)
    ax.set_xticklabels(('E', 'N', 'A', 'C', 'O'))
    fig.set_facecolor('white')
    axes = plt.gca()
    axes.set_ylim([0, 40])
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), (u'j. holenderski'.encode('utf8'), u'j. włoski'.encode('utf8'),
                                                             u'j. angielski', u'j. hiszpański'.encode('utf8')))
    plt.savefig(filename + '.png')