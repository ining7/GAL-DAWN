#!/home/lxr/anaconda3/bin/python
import pandas as pd
import matplotlib.pyplot as plt

# 从xlsx文件中读取数据
df = pd.read_excel('/home/lxr/code/input/gunrock.xlsx')

# 选择需要的列
df = df[['graph', 'SOVM', 'BFS', 'GAP']]

# 设置图名为横坐标
x = df['graph']

# 设置需要显示的折线数据
y1 = df['SOVM']
y2 = df['BFS']
y3 = df['GAP']

# 绘制折线图
plt.figure(figsize=(10, 6))

plt.plot(x, y1, '--', color='red', marker='o', markerfacecolor="white",
         markersize=2.5, label='SOVM', linewidth=0.5)
plt.plot(x, y2, '--', color='green', marker='^', markerfacecolor="white",
         markersize=2.5, label='BFS', linewidth=0.5)
plt.plot(x, y3, '--', color='blue', marker='s', markerfacecolor="white",
         markersize=2.5, label='GAP', linewidth=0.5)


# 设置对数轴
plt.yscale('log')

# 设置图例
plt.legend()

# 设置标题和标签
plt.title('Line Chart')
plt.xlabel('Graph')
plt.xticks(rotation=90)
plt.ylabel('Value')

plt.savefig('1.png')

# 显示图
plt.show()
