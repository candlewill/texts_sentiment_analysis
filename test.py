import matplotlib.pyplot as plt
from matplotlib.patches import Circle # for simplified usage, import this patch

# set up some x,y coordinates and radii
x = [1.0, 2.0, 4.0]
y = [1.0, 2.0, 2.0]
r = [1/(2.0**0.5), 1/(2.0**0.5), 0.25]

fig = plt.figure()

# initialize axis, important: set the aspect ratio to equal
ax = fig.add_subplot(111, aspect='equal')

# define axis limits for all patches to show
ax.axis([min(x)-1., max(x)+1., min(y)-1., max(y)+1.])

# loop through all triplets of x-,y-coordinates and radius and
# plot a circle for each:
for x, y, r in zip(x, y, r):
    ax.add_artist(Circle(xy=(x, y), 
                  radius=r))

plt.show()