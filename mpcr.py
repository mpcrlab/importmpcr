import matplotlib.pyplot as plt

def plot(x):
  fig, ax = plt.subplots()
  im = ax.imshow(x)
  ax.axis('off')
  fig.set_size_inches(18, 10)
  plt.show()
