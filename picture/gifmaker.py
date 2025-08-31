import imageio
images = []
path = "interval_new"
# filenames = (rf'picture/{path}/1.png', rf'picture/{path}/2.png', rf'picture/{path}/3.png', rf'picture/{path}/4.png', rf'picture/{path}/5.png', rf'picture/{path}/6.png', rf'picture/{path}/7.png')

filenames = (rf'picture/{path}/1.png', rf'picture/{path}/2.png', rf'picture/{path}/3.png', rf'picture/{path}/4.png', rf'picture/{path}/5.png')
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('output.gif', images, duration=0.7)