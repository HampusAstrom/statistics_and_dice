import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# [0, 0, 0, 0] = [suc, adv, tri, dis] or reverse for diff dice
""" Hard coded dice """
B = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0], [0, 1, 0, 0]])
K = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])*-1

G = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0]])
P = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [1, 1, 0, 0]])*-1

Y = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0]])
R = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0], [1, 0, 0, -1]])*-1

DICE = {'b': B, 'g': G, 'y': Y, 'k': K, 'p': P, 'r': R}
SIDES = {'b': 6, 'g': 8, 'y': 12, 'k': 6, 'p': 8, 'r': 12}

# ignore tri/dis but track +/- separately
MAX = {'b': [1, 2, 0, 0], 'g': [2, 2, 0, 0], 'y': [2, 2, 0, 0], 'k': [0, 0, 1, 1], 'p': [0, 0, 2, 2], 'r': [0, 0, 2, 2]}

""" Dice to roll """
b = 0
g = 2
y = 1

k = 0
p = 2
r = 1

num_rolls = 100000

dice = {'b': b, 'g': g, 'y': y, 'k': k, 'p': p, 'r': r}

def roll_dice(dice):
    sum = np.array([0, 0, 0, 0])
    for key, value in dice.items():
        for i in range(value):
            sum += DICE[key][np.random.randint(SIDES[key])]
    return sum


rolls = []
for i in range(num_rolls):
    rolls.append(roll_dice(dice))
rolls = np.array(rolls)
print(rolls)

marg = [0, 0, 0, 0]
for key, value in dice.items():
    marg += np.array(MAX[key])*value

# skip tri/dis for now
dims = marg[:2]+[1, 1]+marg[2:]
print(dims)
print(marg)
print()


""" Bin data """
num = np.zeros(dims)
#dims_col = list(dims) + [3]
#col = np.zeros(dims_col)
tri = np.zeros(dims)
dis = np.zeros(dims)
for res in rolls:
    num[res[0]+marg[2]][res[1]+marg[3]] += 1
    tri[res[0]+marg[2]][res[1]+marg[3]] += res[2]
    dis[res[0]+marg[2]][res[1]+marg[3]] += res[3]
    #col[res[0]+marg[2]][res[1]+marg[3]] += [res[3], res[2], 0]
num /= num_rolls
#col /= num_rolls
#rmax = np.max(col[:,:,0])
#gmax = np.max(col[:,:,1])
#print(rmax)
#print(gmax)
tri = np.nan_to_num(tri/num)
dis = np.nan_to_num(dis/num)

""" Plotting """
y = np.linspace(-marg[2], marg[0], marg[0]+marg[2]+1)
x = np.linspace(-marg[3], marg[1], marg[1]+marg[3]+1)

xv, yv = np.meshgrid(x, y)
#print(xv)
#print(yv)

fig, ax = plt.subplots(2,2)#, subplot_kw={"projection": "3d"})
#ax.plot_surface(xv, yv, num, edgecolor='none')
ax[0, 0].pcolor(xv, yv, num, cmap='Greys')
ax[0, 0].set_title('Density')
ax[0, 0].axhline(0, color='blue', lw=1)
ax[0, 0].axvline(0, color='blue', lw=1)
ax[0, 1].pcolor(xv, yv, tri, cmap='Greens')
ax[0, 1].set_title('Triumph')
ax[0, 1].axhline(0, color='blue', lw=1)
ax[0, 1].axvline(0, color='blue', lw=1)
ax[1, 0].pcolor(xv, yv, dis, cmap='Reds')
ax[1, 0].set_title('Dispair')
ax[1, 0].axhline(0, color='blue', lw=1)
ax[1, 0].axvline(0, color='blue', lw=1)
#ax.pcolormesh(xv, yv, num, shading='gouraud', cmap=cmap)

for a in ax.flat:
    a.set(xlabel='advantage', ylabel='success')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for a in ax.flat:
    a.label_outer()
plt.show()
