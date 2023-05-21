import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter

# 自定义格式化函数


def format_func(value, tick_number):
    val = int(value)
    if val % 2 == 0:
        return f"{val}"
    else:
        return ""


# 生成数据
# 64线程 分组输入各线程计算时间
group1 = np.array([])
group2 = np.array([])
group3 = np.array([])
group4 = np.array([])
bench = np.array([])
# # 20线程 分组输入各线程计算时间
# group1 = np.array([])
# group2 = np.array([])
# group3 = np.array([])
# group4 = np.array([])
# bench = np.array([])

for i in range(len(bench)):
    group1[i] = bench[i] / group1[i]
    group2[i] = bench[i] / group2[i]
    group3[i] = bench[i] / group3[i]
    group4[i] = bench[i] / group4[i]

# 每组数据的名称
labels = np.array(['loc-Brightkite', 'mouse_gene', 'mycielskian16',
                   'p2p-Gnutella31', 'soc-sign-Slashdot081106', 'soc-sign-Slashdot090216',
                   'soc-sign-Slashdot090221', 'soc-Slashdot0902'])

# 绘制条形图
fig, ax = plt.subplots()
index = np.arange(len(labels))
bar_width = 0.2
opacity = 0.8

# 设置x轴刻度线的位置和间隔
major_locator = MultipleLocator(1.0)
minor_locator = FixedLocator([2, 4, 8])
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)

# 设置x轴刻度线的显示格式
ax.xaxis.set_major_formatter(FuncFormatter(format_func))


# rects1 = ax.barh(index, group1, bar_width,
#                  alpha=opacity, color='y', label='3 threads')
# rects2 = ax.barh(index + bar_width, group2, bar_width,
#                  alpha=opacity, color='b', label='6 threads')
# rects3 = ax.barh(index + bar_width * 2, group3, bar_width,
#                  alpha=opacity, color='g', label='12 threads')
# rects4 = ax.barh(index + bar_width * 3, group4, bar_width,
#                  alpha=opacity, color='c', label='20 threads')

rects1 = ax.barh(index, group1, bar_width,
                 alpha=opacity, color='y', label='8 threads')
rects2 = ax.barh(index + bar_width, group2, bar_width,
                 alpha=opacity, color='b', label='16 threads')
rects3 = ax.barh(index + bar_width * 2, group3, bar_width,
                 alpha=opacity, color='g', label='32 threads')
rects4 = ax.barh(index + bar_width * 3, group4, bar_width,
                 alpha=opacity, color='c', label='64 threads')

ax.set_xlabel('Speedup')
ax.set_yticks(index + bar_width)
ax.set_yticklabels(labels)

# 将图例放置在图外并设置一行中最大的列数
legend = ax.legend(bbox_to_anchor=(-0.54, 1.1), loc='upper left', ncol=4)

# 显示网格线
plt.grid(linestyle='--', alpha=0.5, axis="x")


# 显示图形
plt.tight_layout()
plt.savefig('scalability64.png')
plt.show()
