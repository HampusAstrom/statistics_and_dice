import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

start_capital = 800000
per_month = 8000

yearly_factor = 1.05
monthly_factor = yearly_factor**(1/12)

years = 20


log = [start_capital]
intrest = []

for y in range(years):
    for m in range(12):
        temp = (log[-1]+per_month)*monthly_factor
        intrest.append(temp-log[-1]-per_month)
        log.append(temp)

log = log[1:]

x = np.linspace(0, years, years*12)

fig, ax = plt.subplots()
ax.plot(x, log)
ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax2=ax.twinx()
ax2.plot(x, intrest, ':r')
plt.show()
