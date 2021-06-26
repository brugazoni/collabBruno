# from https://stackoverflow.com/questions/18344934/animate-a-rotating-3d-graph-in-matplotlib
# recommended tutorial: http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

ECO_SEED = 23

saveit = True

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
  names = np.array(['2S7RApTsKT0CtYojYq2c.{0}'.format(i) for i in range(n)])

  # creates a figure and a 3D Axes
  fig = plt.figure()
  panel1 = Axes3D(fig)

  sc = panel1.scatter(xx, yy, zz, marker='o', s=5, c="blue", alpha=0.05)
  i2si = np.random.randint(n, size=4) # index to sample index
  for i in range(3):
    si = i2si[i]
    v = (xx[si], yy[si], zz[si])
    for j in range(i+1, 4):
      sj = i2si[j]
      w = (xx[sj], yy[sj], zz[sj])
      A = np.vstack((v,w)).T
      panel1.plot(A[0], A[1], A[2], 'r-')

  if(saveit):

    # runs the animation
    movements   = [360,  60, 360,  60, 60, 360, 61]
    transitions = [sum(movements[:i+1]) for i in range(len(movements))]
    numOfFrames = sum(movements)
    # [360, 420, 780, 840, 900, 1260, 1320]

    def rotate_azim(i):

      if(i < transitions[0]):
        (elev, azim) = (0., i)  # cw  rotation on Z
      elif(i < transitions[1]):
        (elev, azim) = (i, 0.)  # cw  rotation on Y
        elev -= transitions[0]
      elif(i < transitions[2]):
        (elev, azim) = (movements[1], i) # ccw rotation on Z
        azim -= transitions[2]
        azim *= -1
      elif(i < transitions[3]):
        (elev, azim) = (i, 0.)  # ccw rotation on Y
        elev -= transitions[3]
        elev *= -1
      elif(i < transitions[4]):
        (elev, azim) = (i, 0.)  # ccw rotation on Y
        elev -= transitions[4]
        elev  = -(movements[4] + elev)
      elif(i < transitions[5]):
        (elev, azim) = (-movements[4], i) # ccw rotation on Z
        azim -= transitions[5]
        azim  = -(movements[5] + azim)
      elif(i <= transitions[6]):
        (elev, azim) = (i, 0.)  # cw  rotation on Y
        elev -= transitions[6]

      else:
        (elev, azim) = (0., 0.)


      print('{0}\t{1}\t{2}'.format(i, elev, azim))
      panel1.view_init(elev=elev, azim=azim)
      return fig,

    #    cw rotation on Z      cw rotation on Y      ccw rotation on Z     ccw on Y
    # [---------------------[---------------------[---------------------[---------------------]
    # 0                    360                   420                   780                   840

    anim = animation.FuncAnimation(fig, rotate_azim, init_func=None,
                                   frames=numOfFrames, interval=200, blit=True)
                                   #frames=720, interval=200, blit=True)
                                   #frames=641, interval=200, blit=True)

    # saves the animation
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

  else:

    annot = panel1.annotate('', xy=(0,0), xytext=(20,20), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))

    annot.set_visible(False)

    def update_annot(ind):

      pos  = sc.get_offsets()[ind['ind'][0]]
      text = '{0}: {1}'.format(ind['ind'][0], names[ind['ind'][0]])
      annot.xy = pos
      annot.set_text(text)
      annot.get_bbox_patch().set_alpha(0.6)

    def hover(event):
      vis = annot.get_visible()
      if event.inaxes == panel1:
        cont, ind = sc.contains(event)
        print('cont: {0}'.format(cont))
        print('ind : {0}'.format(ind))
        if cont:
          update_annot(ind)
          annot.set_visible(True)
          fig.canvas.draw_idle()
        else:
          if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

if(__name__ == '__main__'):

  main()
