"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 5
setrain_means = (0.9297, 0.92515, 0.8038, 0.92855, 0.81705)
setrain_std = (0.0009, 0.00125, 0.0015, 0.00175, 0.00135)


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, setrain_means, width, color='r', yerr=setrain_std)

setest_means = (0.68855, 0.5495, 0.5091, 0.6752, 0.42805)
setest_std = (0.07545, 0.1147, 0.1551, 0.0329, 0.14855)
rects2 = ax.bar(ind + width, setest_means, width, color='y', yerr=setest_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy and Std')
ax.set_title('Accuracy and Std for different classifiers')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('RBF', 'Poly', 'Linear', 'KNN', 'MLP'))

ax.legend((rects1[0], rects2[0]), ('Training', 'Test'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        #ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

# import numpy as np
# rbf = [0.2795, 0.5766]
# mean = np.mean(rbf)
# std = np.std(rbf)
# print('mean', mean)
# print('std', std)