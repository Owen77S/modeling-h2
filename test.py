import matplotlib.pyplot as plt
import optimization as m
import numpy as np

plant = m.empty_plant(104)

WP = plant.data["WP"]
NP = plant.data["NP"]
x = np.arange(0, plant.duration, 1)
y = [plant.data["NP"], plant.data["WP"]]

fig, ax = plt.subplots()

ax.plot(x, WP, 'orange')
ax.set_title("Hourly yield power from the wind plant")
ax.set_xlabel("Hour")
ax.set_ylabel("Power [kW]")

fig.show()

fig, ax = plt.subplots()

ax.plot(x, NP)
ax.set_title("Hourly yield power from the nuclear plant")
ax.set_xlabel("Hour")
ax.set_ylabel("Power [kW]")

fig.show()

gl = np.mean(WP)+np.mean(NP)

fig, ax = plt.subplots()

ax.stackplot(x, y, labels=["Nuclear power", "Wind power"])
ax.plot(x, [gl]*len(x), '--', label="Grid limit")

ax.set_title("Hourly yield power from the two power plants, and the grid limit")
ax.set_xlabel("Hour")
ax.set_ylabel("Power [kW]")

ax.legend()

fig.show()


EP = [max(0, wp+np-gl) for wp, np in zip(WP, NP)]
fig, ax = plt.subplots()


ax.plot(x, EP, 'green')

ax.set_title("Hourly excess power supplied to the H2 plant")
ax.set_xlabel("Hour")
ax.set_ylabel("Power [kW]")

fig.show()