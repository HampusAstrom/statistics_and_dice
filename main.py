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

mu = 3
#variance = 30
sigma = 10
samples = 10


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

print("Probability to succed estimated to: {}".format(succ_rate_samp))
print("Real success probability:           {}".format(succ_rate_real))

plt.plot(x, dist)
plt.plot(x, dist2)
plt.show()
