import matplotlib.pyplot as plt
import random as rng

def get_numbers(num, mean, std_dev):
    ret = [0] * num
    for i in range(num):
        ret[i] = int(round(rng.gauss(mean, std_dev)))
    return ret

def bins(array):
    mini = min(array)
    maxi = max(array)
    ret = [0] * (maxi + 1 - mini)
    for i in range(len(array)):
        ret[array[i] - mini] += 1
    return ret, mini, maxi

num = 1000000
a, an, ax = bins(get_numbers(num, 0, 10))
ascale = [i for i in range(an, ax + 1)]

b, bn, bx = bins(get_numbers(num, 1, 10))
bscale = [i for i in range(bn, bx + 1)]

a = [x / (num * 1.0) for x in a]
b = [x / (num * 1.0) for x in b]

dn = min(an, bn)
dx = max(ax, bx)
diffab = [0] * (dx + 1 - dn)
diffscale = [i for i in range(dn, dx + 1)]
for i in range(dn, dx + 1):
    if i >= an and i <= ax:
        diffab[i] += a[i-an]
    if i >= bn and i <= bx:
        diffab[i] -= b[i-bn]

plt.plot(ascale, a)
plt.plot(bscale, b)
#plt.show()

plt.plot(diffscale, diffab)
plt.show()
