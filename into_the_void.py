import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import sys

def estimate_succ_rate(x, dist):
    succ = 0
    fail = 0
    for inp, den in zip(x, dist):
        if inp < 0:
            fail += den
        else:
            succ += den
    return succ/(fail + succ)

def comb_dist_val(dist_one, dist_two, max_size):
    summed = [0]*max_size
    for i in range(len(dist_one)):
        for j in range(len(dist_two)):
            prob = dist_one[i]*dist_two[j]
            if prob > 0:
                sum = i + j
                summed[sum] += prob
    return summed

def dice_dist():
    sides = 10
    max_dice = 10
    max_size = sides*max_dice+1

    one_die = [0]
    one_die.extend([1.0/sides]*sides)
    one_die.extend([0]*(max_size-sides-1))
    #print(one_die)
    dice_dist = [one_die]

    for i in range(max_dice-1):
        dice_dist.append(comb_dist_val(dice_dist[i], one_die, max_size))
        plt.plot(dice_dist[i])
    plt.show()

def plot_prep(start, stop, ymax, ax):
    ax.plot([0, 0], [0, ymax], 'k', zorder=11)
    ax.fill_between([start, 0], [ymax, ymax], y2=[0, 0], color='k', zorder=12, alpha=0.2)
    plt.ylim(0, ymax)
    plt.xlim(start, stop)

def plot_dist(x, dist, style, order, ax):
    y_zero = [0 for val in x]
    ax.plot(x, dist, style)
    ax.fill_between(x, dist, color=style, y2=y_zero, zorder=order, alpha=0.2)

def sample_dist(x, mu, sigma, samples, func):
    samp = func.rvs(loc=mu, scale=sigma, size=samples)

    mean = np.mean(samp)
    std = np.std(samp)

    if std <= 0.0:
        std = 10/(np.sqrt(0.5+1))

    dist = func.pdf(x, mean, std)
    return dist

def dist_and_sample(mu, sigma, samples, start='default', stop='default'):
    if start == 'default':
        start = mu - 3*sigma
    if stop == 'default':
        stop = mu + 3*sigma

    #func = stats.crystalball
    #func = stats.gumbel_l
    func = stats.norm

    x = np.linspace(start, stop, 100)
    dist = func.pdf(x=x, loc=mu, scale=sigma)

    dist2 = sample_dist(x, mu, sigma, samples, func)

    #print(sum(dist)*(stop-start)/len(x))
    succ_rate_samp = estimate_succ_rate(x, dist2)
    succ_rate_real = estimate_succ_rate(x, dist)

    print("Probability to succeed estimated to: {}".format(succ_rate_samp))
    print("Real success probability:            {}".format(succ_rate_real))

    fig, ax = plt.subplots()

    ymax = 0.4
    plot_prep(start, stop, ymax, ax)

    plot_dist(x, dist, 'b', 9, ax)
    plot_dist(x, dist2, 'r', 10, ax)
    plt.show()

def plot_multi_dist(mu, sigma, start=-10, stop=10):
    func = stats.norm
    x = np.linspace(start, stop, 100)
    fig, ax = plt.subplots()

    ymax = 0.4
    plot_prep(start, stop, ymax, ax)


    plot_prep(start, stop, ymax, ax)
    for i in range(len(mu)):
        dist = func.pdf(x=x, loc=mu[i], scale=sigma[i])
        plot_dist(x, dist, 'b', i, ax)

    plt.show()

    fig, ax = plt.subplots()
    ymax = 1
    plot_prep(start, stop, ymax, ax)

    for i in range(len(mu)):
        dist = func.pdf(x=x, loc=mu[i], scale=sigma[i])
        sum_dist = dist
        summed = 0
        stepsize = (stop - start)/100
        for i in range(len(sum_dist)-1, -1, -1):
            summed += dist[i]*stepsize
            sum_dist[i] = summed
        plot_dist(x, sum_dist, 'b', i, ax)

    plt.show()

def sigma_for_check(attibute, skill):
    if skill > 0:
        sigma = 10/(np.sqrt(skill+1))
    else:
        sigma = 10/(np.sqrt(0.5+1))
    return sigma

# maybe mu is defined by difference (attr + skill - difficulty)
# how is "real" sigma defined?
# skill sigma goes something like 1/sqrt(attr + skill) or 1/sqrt(skill)
# maybe also some diff sigma that varies independently, thou prob not
# hard to estimate sigma is still applied to estimation check only, not real check (sometimes)
# sample size can be determined by (insight + skill)

# check other distributions, probably with more tail towards fail than succeed

# let attr and skill scale by fibonacchi sequence? (from different pools prob)
# they go from like 1-6 each. 1 1 2 3 5 8

mu = 3
sigma = 5
samples = 5

#dist_and_sample(mu, sigma, samples)
#dice_dist()

def main(argv):
    if len(argv) < 4:
        print("Needs 4 arguments, attribute, insight, skill, diff, (bonus optional)")
        exit()

    attr = int(argv[0]) #3
    insight = int(argv[1]) #2
    skill = int(argv[2]) #3

    diff = int(argv[3]) #6

    if len(argv) > 4:
        bonus = int(argv[4]) # 0
    else:
        bonus = 0

    mu = attr + skill + bonus - diff
    sigma = sigma_for_check(attr, skill)
    samples = insight + skill
    if samples < 1:
        samples = 1

    dist_and_sample(mu, sigma, samples)
    func = stats.norm
    a = func.rvs(loc=mu, scale=sigma, size=1)[0]
    if a >= 0:
        print(f"If this was the roll, you succeded with {a}")
    else:
        print(f"If this was the roll, you failed with {a}")

# top_val = 7
# m = [0]*top_val
# s = [0]*top_val
# for attri in range(2,3):
#     for ski in range(top_val):
#         m[ski] = attri + ski - diff
#         s[ski] = sigma_for_check(attri, ski)
#     plot_multi_dist(m, s)

if __name__ == "__main__":
   main(sys.argv[1:])
