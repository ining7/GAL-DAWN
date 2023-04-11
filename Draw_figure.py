import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
x = ['bmw7st_1', 'bmwcra_1', 'boneS01', 'crankseg_1', 'crankseg_2', 'fullb', 'Ga10As10H30', 'Ga19As19H42', 'Ge99H100', 'kron_g500-logn17',
     'loc-Brightkite', 'm_t1', 'mono_500Hz', 'mouse_gene', 'mycielskian16', 'mycielskian17', 'mycielskian18', 'nd24k', 'p2p-Gnutella31', 'para-8',
     'pkustk14', 'PR02R', 'Si41Ge41H72', 'SiO2', 'soc-sign-epinions', 'soc-sign-Slashdot081106', 'soc-sign-Slashdot090216', 'soc-sign-Slashdot090221',
     'soc-Slashdot0902', 'sx-askubuntu', 'sx-superuser', 'torso1', 'TSOPF_FS_b300_c3', 'wave', 'x104', 'xenon2']


# Speedup
# Fill in the relevant data in the order of the graph name (Order: From A to Z)
y1 = []  # Connection
y2 = []  # Ndoes
y3 = []  # Speedup
y4 = []  # Diameter

# Divide the data in the two lists by the minimum value in the list
y1 = np.log(np.array(y1)/np.min(y1))  # Connection
y2 = np.log(np.array(y2)/np.min(y2))  # Ndoes
y3 = np.log(np.array(y3)/np.min(y3))  # Speedup
y4 = np.log(np.array(y4)/np.min(y4))  # Diameter

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

fig.savefig('figure.png')


# Latency
# Fill in the relevant data in the order of the graph name (Order: From A to Z)
y6 = []  # GDS Latency
y7 = []  # DAWN Latency

y6 = np.log(np.array(y6)/np.min(y6))  # GDS Latency
y7 = np.log(np.array(y7)/np.min(y7))  # DAWN Latency

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

# Plot the line chart on the subplot
ax1.plot(x, y6, '--', color='blue', marker='o',
         markerfacecolor="white", markersize=6, label='Latency of GDS')
ax1.plot(x, y1, '--', color='green', marker='^', markerfacecolor="white",
         markersize=6, label='Probability of Average Connection')
ax2.plot(x, y7, '--', color='red', marker='s',
         markerfacecolor="white", markersize=6, label='Latency of DAWN')
ax2.plot(x, y2, '--', color='black', marker='p',
         markerfacecolor="white", markersize=6, label='Number of Nodes')
ax2.plot(x, y4, '--', color='brown', marker='*',
         markerfacecolor="white", markersize=5, label='Diameter')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
plt.xticks(rotation=90)

plt.savefig('figure.png')


# Show the plot
plt.show()
