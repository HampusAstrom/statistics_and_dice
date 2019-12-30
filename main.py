import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def estimate_succ_rate(x, dist):
    succ = 0
    fail = 0
    for inp, den in zip(x, dist):
        if inp < 0:
            fail += den
        else:
            succ += den
    return succ/(fail + succ)

def comb_dist_val(dist_one, dist_two):
    summed = [0]*max_size
    for i in range(len(dist_one)):
        for j in range(len(dist_two)):
            prob = dist_one[i]*dist_two[j]
            if prob > 0:
                sum = i + j
                summed[sum] += prob
    return summed

# maybe mu is defined by difference (attr + skill - difficulty)
# how is "real" sigma defined?
# skill sigma goes something like sqrt(attr + skill) or sqrt(skill)
# maybe also some diff sigma that varies independently, thou prob not
# hard to estimate sigma is still applied to estimation check only, not real check (sometimes)
# sample size can be determined by (insight + skill)

# check other distributions, probably with more tail towards fail than succeed

mu = 3
#variance = 30
sigma = 5
samples = 5


#sigma = math.sqrt(variance)
start = mu - 3*sigma
stop = mu + 3*sigma
x = np.linspace(start, stop, 100)
dist = stats.norm.pdf(x, mu, sigma)


samp = np.random.normal(mu, sigma, samples)

mean = np.mean(samp)
std = np.std(samp)

dist2 = stats.norm.pdf(x, mean, std)

print(sum(dist)*(stop-start)/len(x))

succ_rate_samp = estimate_succ_rate(x, dist2)
succ_rate_real = estimate_succ_rate(x, dist)

print("Probability to succeed estimated to: {}".format(succ_rate_samp))
print("Real success probability:            {}".format(succ_rate_real))

fig, ax = plt.subplots()

ymax = 0.2

ax.plot([0, 0], [0, ymax], 'k', zorder=11)
y_zero = [0 for val in x]
ax.fill_between([start, 0], [ymax, ymax], y2=[0, 0], color='k', zorder=12, alpha=0.2)
plt.ylim(0, ymax)
plt.xlim(start, stop)
ax.plot(x, dist, 'b')
ax.plot(x, dist2, 'r')
ax.fill_between(x, dist, color='b', y2=y_zero, zorder=9, alpha=0.2)
ax.fill_between(x, dist2, color='r', y2=y_zero, zorder=10, alpha=0.2)
plt.show()

sides = 10
max_size = 150

one_die = [0]
one_die.extend([1.0/sides]*sides)
one_die.extend([0]*(max_size-sides-1))
print(one_die)
dice_dist = [one_die]

for i in range(10):
    dice_dist.append(comb_dist_val(dice_dist[i], one_die))
    plt.plot(dice_dist[i])
plt.show()
