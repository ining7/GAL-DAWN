import matplotlib.pyplot as plt
import numpy as np

# 数据
Graph = ['bmw7st_1', 'bmwcra_1', 'boneS01', 'crankseg_1', 'crankseg_2', 'fullb', 'Ga10As10H30', 'Ga19As19H42',
         'Ge99H100', 'kron_g500-logn17',
         'loc-Brightkite', 'm_t1', 'mono_500Hz', 'mouse_gene', 'mycielskian16', 'mycielskian17', 'mycielskian18',
         'nd24k', 'p2p-Gnutella31', 'para-8',
         'pkustk14', 'PR02R', 'Si41Ge41H72', 'SiO2', 'soc-sign-epinions', 'soc-sign-Slashdot081106',
         'soc-sign-Slashdot090216', 'soc-sign-Slashdot090221',
         'soc-Slashdot0902', 'sx-askubuntu', 'sx-superuser', 'torso1', 'TSOPF_FS_b300_c3', 'wave', 'x104', 'xenon2']

SSSP = []
BFS = []
DAWN20 = []
DAWN64 = []
DAWN = []
node = [141347, 148770, 127224, 52804, 63838, 199187, 113081, 133123, 112985,
        131072, 58228, 97578, 169410, 45101, 49151, 98303, 196607, 72000, 62586, 155924,
        151926, 161070, 185639, 155331, 131828, 77350, 81867, 82140, 82168, 159316,
        194085, 116158, 84414, 156317, 108384, 157464]
Connection = [0.000187222172842, 0.000243821658407, 0.000211367810247, 0.001912842553626, 0.001743764879192,
              0.000150058294336, 0.000243550549433, 0.000254432281943, 0.000335447340513, 0.000297695805784,
              0.000063140540041, 0.000517312560539, 0.000175481958324, 0.007131505120971, 0.006909138290841,
              0.005186834555531, 0.003892629475651, 0.002776585069444, 0.000037756374969, 0.000222782681466,
              0.000324684787526, 0.000315497979615, 0.000220488876108, 0.000237047557915, 0.000048414196485,
              0.000086340139954, 0.000081416629886, 0.000081399682904, 0.000140480297728, 0.000037997504548,
              0.000038316373928, 0.000631194054223, 0.000923240018507, 0.000043353010247, 0.000437385204605,
              0.000155946837391]

# Diameter
Diameter = [99, 98, 49, 28, 28, 83, 16, 17, 15, 12,
            11, 48, 41, 2, 2, 2, 2, 14, 11, 44,
            14, 137, 17, 17, 14, 11, 11, 12, 12,
            13, 12, 126, 14, 56, 86, 14]

# Latency
# Fill in the relevant data in the order of the graph name (Order: From A to Z)
y1 = np.array(BFS) / np.array(node)
y2 = np.array(DAWN) / np.array(node)
y3 = np.array(DAWN20) / np.array(node)
y4 = np.array(DAWN64) / np.array(node)
y5 = np.array(SSSP) / np.array(node)

# Plot the line chart on the subplot
plt.plot(Graph, y5, '--', color='blue', marker='o', markerfacecolor="white",
         markersize=2.5, label='Latency of GDS', linewidth=0.5)
plt.plot(Graph, y1, '--', color='green', marker='^', markerfacecolor="white",
         markersize=2.5, label='Latency of BFS', linewidth=0.5)
plt.plot(Graph, y2, '--', color='red', marker='s', markerfacecolor="white",
         markersize=2.5, label='Latency of DAWN', linewidth=0.5)
plt.plot(Graph, y3, '--', color='black', marker='p', markerfacecolor="white",
         markersize=2.5, label='Latency of DAWN(20)', linewidth=0.5)
plt.plot(Graph, y4, '--', color='c', marker='*', markerfacecolor="white",
         markersize=2.5, label='Latency of DAWN(64)', linewidth=0.5)
plt.yscale('log', base=10)
plt.legend(loc='upper left')
plt.xticks(rotation=90)
plt.savefig('Latency.png')

# Show the plot
plt.show()
