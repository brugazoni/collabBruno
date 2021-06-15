# from https://stackoverflow.com/questions/18344934/animate-a-rotating-3d-graph-in-matplotlib
# recommended tutorial: http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

ECO_SEED = 23

# creates a mock dataset
def randrange(n, vmin, vmax):
  return (vmax - vmin) * np.random.rand(n) + vmin

def main():

  # locks random number generator
  np.random.seed(ECO_SEED)

  # creates a mock dataset
  n = 100
  xx = randrange(n, 23, 32)
  yy = randrange(n, 0, 100)
  zz = randrange(n, -50, -25)

  # creates a figure and a 3D Axes
  fig = plt.figure()
  panel1 = Axes3D(fig)

  # runs the animation
  def init():
    panel1.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
    i2si = np.random.randint(n, size=4) # index to sample index
    for i in range(3):
      si = i2si[i]
      v = (xx[si], yy[si], zz[si])
      for j in range(i+1, 4):
        sj = i2si[j]
        w = (xx[sj], yy[sj], zz[sj])
        A = np.vstack((v,w)).T
        panel1.plot(A[0], A[1], A[2], 'r-')

    return fig,

  def rotate_azim(i):
    print(i)
    if(i < 360):
      panel1.view_init(elev=10., azim=i)
    else:
      #panel1.view_init(azim=1., elev=i-350)
      panel1.view_init(azim=1., elev=370-i)
    return fig,

  anim = animation.FuncAnimation(fig, rotate_azim, init_func=init,
                                 frames=641, interval=200, blit=True)

  # saves the animation
  anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

if(__name__ == '__main__'):

  main()
