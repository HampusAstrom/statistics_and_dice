import itertools
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict

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
        #results.append((numpy.prod(probs), func(values)))
        results[func(values)] += numpy.prod(probs)

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
probs, values = get_dist([Die(6), Die(6), Die(6), Die(6)], lambda x : sum(sorted(x)[1:]))
print(probs)
print(values)
print(average_outcome(probs, values))

plt.bar(values, probs)
plt.show()
