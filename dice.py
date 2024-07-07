import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import sys

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
    0 values denote regular fail (that can still succed if total over target)
    """
    @staticmethod
    def roll(n=1, botch_dice=1):
        arr = rng.integers(10, size=n)
        if 1 in arr:
            arr = Ars_dice._roll_forward(arr, multiplier=2)
        return Ars_dice._resolve_botches(arr, botch_dice=botch_dice)

DEBUG = False

def enchantment_rite(char, task, samples=1):
    curr_magnitude = task["starting_magnitude"] -1
    skill_total = char["characteristic"] + char["performance_skill"] \
                + char["highest_sympathy"]
    rite_total = char["characteristic"] + char["method"] + char["power"] \
                + char["aura"]
    bonus = np.zeros(samples)
    failures = np.empty(samples)
    failures[:] = np.nan
    while curr_magnitude < task["final_magnitude"]:
        curr_magnitude += 1
        rolls = Ars_dice.roll(samples, botch_dice=task["skill_botch_dice"])
        # track new botches
        new_botched = np.logical_and(np.isnan(failures), rolls < 0)
        failures[new_botched] = rolls[new_botched]

        # compute results
        results = rolls + bonus + skill_total - curr_magnitude*3
        # track new failures
        new_failed = np.logical_and(np.isnan(failures), results < 0)
        failures[new_failed] = 0

        # save bonus for next round (dont care about what happens to failed
        # rolls, their results are ignored in the future)
        bonus = results
        if DEBUG:
            print(f"{rolls} rolls")
            print(f"{results} results")
            print(f"{failures} failures")


    rolls = Ars_dice.roll(samples, botch_dice=task["rite_botch_dice"])
    rite_results = rolls + bonus + rite_total - curr_magnitude*5

    rite_failures = np.empty(samples)
    rite_failures[:] = np.nan

    # mark failiours as 0
    rite_failures[rite_results < 0] = 0

    # mark rite botches as -x from degree of botch in roll
    rite_failures[rolls < 0] = rolls[rolls < 0]

    # remove all that failed earlier
    rite_results[~np.isnan(failures)] = np.nan

    # remove all that failed now
    rite_results[~np.isnan(rite_failures)] = np.nan
    if DEBUG:
        print(f"{rolls} rite rolls")
        print()

    return rite_results, failures, rite_failures



# rolls = Ars_dice.roll(10, botch_dice=2)
# print(rolls)

# Enchantment (beguile) test
# rite total = characteristic + method + power + aura + die
# performance total = characteristic (often same) + ability + die + highest sympathy?
# max rite magnitude = sum(absolute val of applicable sympathies)
# rite level = 5 * rite magnitude
# ability diff for final level = 3 * rite magnitude
# succeed on skill first, then rite, but skill overshoot becomes bonus to rite
# can start on lower magnitude for skill, but need to step up one at a time then
# overshoot for previous skill step becomes bonus for next
char = {"characteristic": 5,
        "method": 3,
        "power": 3,
        "aura": 0,
        "performance_skill": 6,
        "highest_sympathy": 0,}
task = {"final_magnitude": 6,
        "starting_magnitude": 1,
        "skill_botch_dice": 2,
        "rite_botch_dice": 2,}
samples = 10000

rite_results, failures, rite_failures = enchantment_rite(char, task, samples)
# print(f"{rite_results} rite results")
# print(f"{failures} skill failures")

rite_res = rite_results[~np.isnan(rite_results)]
fails = failures[~np.isnan(failures)]
rite_fails = rite_failures[~np.isnan(rite_failures)]

perf_botches = np.sum(fails < 0)
rite_botches = np.sum(rite_fails < 0)

unique_res, result_counts = np.unique(rite_res, return_counts=True)
unique_fail, failure_counts = np.unique(fails, return_counts=True)
unique_rite_fail, rite_failure_counts = np.unique(rite_fails, return_counts=True)

if 0 not in unique_fail:
    unique_fail = np.append(unique_fail,0)
    failure_counts = np.append(failure_counts,0)

if 0 not in unique_rite_fail:
    unique_rite_fail = np.append(unique_rite_fail,0)
    rite_failure_counts = np.append(rite_failure_counts,0)

print(unique_res)
print(result_counts)
print(unique_fail)
print(failure_counts)
print(unique_rite_fail)
print(rite_failure_counts)

print(f"Success rate: {len(rite_res)/samples}")
print(f"Average penetration of successes: {np.sum(rite_res)/len(rite_res)}")
print(f"Total botch rate {(perf_botches+rite_botches)/samples}")
print(f"Performance botch rate {perf_botches/samples}")
print(f"Average performance botch value {np.sum(fails)/perf_botches}")
print(f"Rite botche rate {rite_botches/samples}")
print(f"Average rite botch value {np.sum(rite_fails)/rite_botches}")

sys.stdout.flush()

fig, axs = plt.subplots(2)

axs[0].fill_between(unique_res, result_counts/samples)
axs[1].fill_between(unique_fail, failure_counts/samples)
axs[1].fill_between(unique_rite_fail, (rite_failure_counts+failure_counts)/samples, failure_counts/samples)

plt.show()