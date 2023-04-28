import matplotlib.pyplot as plt
import numpy as np

# We have reused the drawing code, please alternately annotate the elements of the two pictures when using it.

# Generate some random data
x = ['bmw7st_1', 'bmwcra_1', 'boneS01', 'crankseg_1', 'crankseg_2', 'fullb', 'Ga10As10H30', 'Ga19As19H42', 'Ge99H100', 'kron_g500-logn17',
     'loc-Brightkite', 'm_t1', 'mono_500Hz', 'mouse_gene', 'mycielskian16', 'mycielskian17', 'mycielskian18', 'nd24k', 'p2p-Gnutella31', 'para-8',
     'pkustk14', 'PR02R', 'Si41Ge41H72', 'SiO2', 'soc-sign-epinions', 'soc-sign-Slashdot081106', 'soc-sign-Slashdot090216', 'soc-sign-Slashdot090221',
     'soc-Slashdot0902', 'sx-askubuntu', 'sx-superuser', 'torso1', 'TSOPF_FS_b300_c3', 'wave', 'x104', 'xenon2']


# Speedup
# Fill in the relevant data in the order of the graph name (Order: From A to Z)
# Connection
y1 = [0.000187222172842, 0.000243821658407, 0.000211367810247, 0.001912842553626, 0.001743764879192, 0.000150058294336, 0.000243550549433, 0.000254432281943, 0.000335447340513, 0.000297695805784,
      0.000063140540041, 0.000517312560539, 0.000175481958324, 0.007131505120971, 0.006909138290841, 0.005186834555531, 0.003892629475651, 0.002776585069444, 0.000037756374969, 0.000222782681466,
      0.000324684787526, 0.000315497979615, 0.000220488876108, 0.000237047557915, 0.000048414196485, 0.000086340139954, 0.000081416629886, 0.000081399682904, 0.000140480297728, 0.000037997504548,
      0.000038316373928, 0.000631194054223, 0.000923240018507, 0.000043353010247, 0.000437385204605, 0.000155946837391]
# Ndoes
y2 = [141347, 148770, 127224, 52804, 63838, 199187, 113081, 133123, 112985,
      131072, 58228, 97578, 169410, 45101, 49151, 98303, 196607, 72000, 62586, 155924,
      151926, 161070, 185639, 155331, 131828, 77350, 81867, 82140, 82168, 159316,
      194085, 116158, 84414, 156317, 108384, 157464]
# Time of Dawn
y3_dawn = []
# Time of gunrock
y3_gunrock = []


# Diameter
y4 = [99, 98, 49, 28, 28, 83, 16, 17, 15, 12,
      11, 48, 41, 2, 2, 2, 2, 14, 11, 44,
      14, 137, 17, 17, 14, 11, 11, 12, 12,
      13, 12, 126, 14, 56, 86, 14]

# Divide the data in the two lists by the minimum value in the list

# Connection
y1 = np.log(np.array(y1)/np.min(y1))

# Ndoes
y2 = np.log(np.array(y2)/np.min(y2))

# Speedup
y3 = np.array(y3_gunrock)/np.array(y3_dawn)
y3 = np.log(np.array(y3)/np.min(y3))

# Diameter
y4 = np.log(np.array(y4)/np.min(y4))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

# Plot the line chart on the subplot
ax1.plot(x, y1, '--', color='green', marker='^', markerfacecolor="white",
         markersize=5, label='Probability of Average Connection')
ax1.plot(x, y2, '--', color='magenta', marker='p',
         markerfacecolor="white", markersize=5, label='Number of Nodes')
ax2.plot(x, y3, '--', color='red', marker='s',
         markerfacecolor="white", markersize=5, label='Speedup')
ax2.plot(x, y4, '--', color='blue', marker='*',
         markerfacecolor="white", markersize=5, label='Diameter')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
plt.xticks(rotation=90)

fig.savefig('Speedup.png')


# Latency
# Fill in the relevant data in the order of the graph name (Order: From A to Z)

# GDS Latency
y6 = np.array(y3_gunrock) / np.array(y2)
y6 = np.log(np.array(y6)/np.min(y6))

# DAWN Latency
y7 = np.array(y3_dawn) / np.array(y2)
y7 = np.log(np.array(y7)/np.min(y7))  # DAWN Latency

# #

# # Plot the line chart on the subplot
# ax1.plot(x, y6, '--', color='blue', marker='o',
#          markerfacecolor="white", markersize=6, label='Latency of GDS')
# ax1.plot(x, y1, '--', color='green', marker='^', markerfacecolor="white",
#          markersize=6, label='Probability of Average Connection')
# ax2.plot(x, y7, '--', color='red', marker='s',
#          markerfacecolor="white", markersize=6, label='Latency of DAWN')
# ax2.plot(x, y2, '--', color='black', marker='p',
#          markerfacecolor="white", markersize=6, label='Number of Nodes')
# ax2.plot(x, y4, '--', color='brown', marker='*',
#          markerfacecolor="white", markersize=5, label='Diameter')


# ax1.legend(loc='upper left')
# ax2.legend(loc='upper left')
# plt.xticks(rotation=90)

# fig.savefig('Latency.png')


# Show the plot
plt.show()

