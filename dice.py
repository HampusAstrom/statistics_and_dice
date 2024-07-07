import numpy as np
from abc import ABC, abstractmethod

rng = np.random.default_rng()

class Dice(ABC):

    @staticmethod
    @abstractmethod
    def roll(n=1):
        return NotImplemented


class Ars_dice(Dice):
    vals = []

    @staticmethod
    def _roll_forward(arr, multiplier):
        filter = arr == 1
        r = rng.integers(1, 11, size=len(arr))
        roll_filter = r == 1
        r[~roll_filter] = r[~roll_filter]*multiplier
        arr[filter] = r[filter]
        if 1 in arr:
            return Ars_dice._roll_forward(arr, multiplier=multiplier*2)
        return arr

    @staticmethod
    def _resolve_botches(arr, botch_dice):
        r = rng.choice([True, False], size=(len(arr), botch_dice), p=[0.1, 0.9])
        botches = r.sum(axis=1)
        filter = arr == 0
        arr[filter] = - botches[filter]
        return arr

    """
    negative values denote size of botch
    positive values is outcome
    0 values denote regular fail
    """
    @staticmethod
    def roll(n=1, botch_dice=1):
        arr = rng.integers(10, size=n)
        if 1 in arr:
            arr = Ars_dice._roll_forward(arr, multiplier=2)
        return Ars_dice._resolve_botches(arr, botch_dice=botch_dice)


rolls = Ars_dice.roll(10, botch_dice=2)
print(rolls)