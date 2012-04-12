import numpy, glumpy, pynbody

fig = glumpy.figure( (512,512) )

try : s
except NameError: 
    s = pynbody.load('/Users/rokstar/Nbody/runs/12M_hr/12M_hr.01000')

Z = numpy.random.random((32,32)).astype(numpy.float32)

image = glumpy.image.Image()

@fig.event
def on_draw():
    fig.clear()
    image.update()
    image.draw( x=0, y=0, z=0, width=fig.width, height=fig.height )

glumpy.show()
