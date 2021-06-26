import matplotlib.pyplot as plt
import numpy as np;

def example2():

  x = np.sort(np.random.rand(15))
  y = np.sort(np.random.rand(15))
  names = np.array(list("ABCDEFGHIJKLMNO"))

  #norm = plt.Normalize(1,4)
  #cmap = plt.cm.RdYlGn

  fig,ax = plt.subplots()
  line, = plt.plot(x,y, marker="o")

  annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
  annot.set_visible(False)

  def update_annot(ind):
    x,y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


  def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
      cont, ind = line.contains(event)
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

def example1():

  ss = 15
  x = np.random.rand(ss)
  y = np.random.rand(ss)
  #names = np.array(list("ABCDEFGHIJKLMNO"))
  names = np.array(['2S7RApTsKT0CtYojYq2c.{0}'.format(i) for i in range(ss)])
  c = np.random.randint(1,5,size=ss)

  #norm = plt.Normalize(1,4)
  #cmap = plt.cm.RdYlGn

  fig,ax = plt.subplots()
  sc = plt.scatter(x, y, c=c, s=100) #, cmap=cmap, norm=norm)

  annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords='offset points',
                      bbox=dict(boxstyle='round', fc='w'),
                      arrowprops=dict(arrowstyle='->'))

  annot.set_visible(False)

  def update_annot(ind):

    pos = sc.get_offsets()[ind['ind'][0]]
    annot.xy = pos
    #text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
    #                       " ".join([names[n] for n in ind["ind"]]))
    text = '{0}: {1}'.format(ind['ind'][0], names[ind['ind'][0]])


    annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.6)


  def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
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

def main():
  np.random.seed(1)
  example1()

if(__name__ == '__main__'):
  main()