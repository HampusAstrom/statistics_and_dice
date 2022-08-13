import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

""" the idea here is to go through all options rather than sample, can be very timeconsuming for some tasks, beware! """

class Die:
    """A class used to represent an die
    Die is iterable

    Attributes
    ----------
    values : int[] or double[]
        an array of the possible outcomes of the die
    probabilities : double[]
        an array of the probabilities of each outcome of the die
        if not initiated each outcome is assumed to be of equal likelyhood

    Methods
    -------

    """

    def __init__(self, sides=6, probabilities=None, *, values=None):
        if probabilities:
            # checks sum of probabilities with some rounding margin
            if sum(probabilities) >= 1.01 or sum(probabilities) <= 0.99:
                raise Exception("Probabilities must sum to 1.0! The provided array sums to {}".format(sum(probabilities)))
            if values:
                if len(values) != len(probabilities):
                    raise Exception("Length of probabilities and values must match!")
                self.probabilities = probabilities
            elif sides != len(probabilities):
                raise Exception("Sides and length of probabilities must match!")
            self.probabilities = probabilities
        else:
            self.probabilities = None

        if values:
            self.values = values
        else:
            self.values = list(range(1,sides+1))

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return self.DieIterator(self)

    class DieIterator:
        def __init__(self, die):
            self.__die = die
            self.__index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.__index >= len(self.__die):
                raise StopIteration

            # return the next outcome, (probability, value)
            if self.__die.probabilities:
                outcome = (self.__die.probabilities[self.__index], self.__die.values[self.__index])
            else:
                outcome = (1/len(self.__die), self.__die.values[self.__index])
            self.__index += 1
            return outcome

def average_outcome(probs, values):
    sum = 0
    sum_p = 0
    for prob, val in zip(probs, values):
        sum += prob*val
        sum_p += prob

    # sometimes it might be closer to correct if divided by total prob (should be 1) due to rounding errors
    #sum /= sum_p
    #print(sum_p)
    return sum

def get_dist(dice, func):
    """ A funtion that finds probabilities of all outcomes for a dice problem

    Parameters
    ----------
    dice : Die[]
        array of Die objects to check outcomes of
    func : function that can operate on array of the same length as dice

    Returns
    -------
    outcomes : tuple[]
        a list of tuples where the first value in each tuple is the probability
        and the second value/rest of each tuble is the outcome

    """
    #results = []
    results = defaultdict(float)
    for outcomes in itertools.product(*dice):
        (probs, values) = zip(*outcomes)
        #results.append((np.prod(probs), func(values)))
        results[func(values)] += np.prod(probs)

    #results = sorted(results, key = lambda x: x[1])
    #probs, values = zip(*results)

    values, probs = zip(*sorted(results.items()))
    return probs, values
    #return results

""" Some testing to wrap propperly later """
# a = Die()
# b = Die(8)
# c = Die(10, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# d = Die(values=[1, 3])
# e = Die(values=[1, 3], probabilities=[0.4, 0.6])
#
# for die in [a, b, c, d, e]:
#     for outcome in die:
#         print(outcome)
#
# print()

#probs, values = get_dist([Die(6), Die(6)], lambda x : sum(x))
#probs, values = get_dist([Die(20), Die(20)], lambda x : max(x))
# probs, values = get_dist([Die(6), Die(6), Die(6), Die(6)], lambda x : sum(sorted(x)[1:]))
# print(probs)
# print(values)
# print(average_outcome(probs, values))
#
# plt.bar(values, probs)
# plt.show()

""" Exploring alternatives for ruling in WoD """
min_pool = 2
max_pool = 10
step_pool = 2
min_will = 2
max_will = 10
step_will = 2

count_ones = True


fig1, ax1 = plt.subplots(int((max_pool-min_pool)/step_pool+1), int((max_will-min_will)/step_will+1), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
#fig1 = plt.figure()

for i in range(min_pool, max_pool+1, step_pool): # i for caster
    print()
    print("i = {}".format(i))
    for j in range(min_will, max_will+1, step_will): # j for target
        print("j = {}".format(j))
        t = time.time()
        # vs roll (vamp)
        if count_ones:
            dice = [Die(3, probabilities=[0.1, 0.4, 0.5], values=[-1, 0, 1]) for x in range(i)]
            dice += [Die(3, probabilities=[0.1, 0.4, 0.5], values=[-1, 0, 1]) for x in range(j)]
        else:
            dice = [Die(2, probabilities=[0.5, 0.5], values=[0, 1]) for x in range(i)]
            dice += [Die(2, probabilities=[0.5, 0.5], values=[0, 1]) for x in range(j)]
        probs, values = get_dist(dice, lambda x : sum(x[:i])-sum(x[i:]))

        ax1[int(i/2)-1, int(j/2)-1].fill_between(values, probs, 0, color='b', alpha=0.2, label='vs vamp')

        # vs roll (mundane)
        if count_ones:
            dice = [Die(3, probabilities=[0.1, 0.4, 0.5], values=[-1, 0, 1]) for x in range(i)]
            dice += [Die(3, probabilities=[0.1, 0.6, 0.3], values=[-1, 0, 1]) for x in range(j)]
        else:
            dice = [Die(2, probabilities=[0.5, 0.5], values=[0, 1]) for x in range(i)]
            dice += [Die(2, probabilities=[0.7, 0.3], values=[0, 1]) for x in range(j)]
        probs, values = get_dist(dice, lambda x : sum(x[:i])-sum(x[i:]))

        ax1[int(i/2)-1, int(j/2)-1].fill_between(values, probs, 0, color='g', alpha=0.2, label='vs mundane')

        # diff roll
        if count_ones:
            dice = [Die(3, probabilities=[0.1, (j-2)*0.1, 1-(j-1)*0.1], values=[-1, 0, 1]) for x in range(i)]
        else:
            dice = [Die(2, probabilities=[(j-1)*0.1, 1-(j-1)*0.1], values=[0, 1]) for x in range(i)]
        probs, values = get_dist(dice, lambda x : sum(x))

        ax1[int(i/2)-1, int(j/2)-1].fill_between(values, probs, 0, color='r', alpha=0.2, label='diff')

        ax1[int(i/2)-1, int(j/2)-1].set_xlim([-(max_pool-1), max_pool+1])
        ax1[int(i/2)-1, int(j/2)-1].set_ylim([0, 0.6])
        ax1[int(i/2)-1, int(j/2)-1].plot([0, 0], [0, 0.4], 'k')
        if j == min_will:
            ax1[int(i/2)-1, int(j/2)-1].set(ylabel='{} dice'.format(i))
        if i == max_pool:
            ax1[int(i/2)-1, int(j/2)-1].set(xlabel='{} dice/diff'.format(j))
            ax1[int(i/2)-1, int(j/2)-1].set_xticks(np.arange(-max_pool+step_pool, max_pool+1, step=step_pool))
        if i == min_pool and j == max_will:
            ax1[int(i/2)-1, int(j/2)-1].legend()
        print("Time used: {}".format(time.time() - t))

for ax in ax1.flat:
    ax.label_outer()

fig1.supxlabel('willpower of target')
fig1.supylabel('discipline dicepool')
plt.show()
