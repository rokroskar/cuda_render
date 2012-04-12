"""

ParticleViz
===========

A lightweight particle vizualization module

"""

import numpy as np
import glumpy as gp
import OpenGL.GL as gl
import pynbody
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import jet


class ParticleViz : 
    """
    
    The basic vizualization class

    """

    
    def __init__(self, sim, mode = 'type') : 
        global fig, V, frame, mesh, trackball

        n = len(sim)
        
        V = np.zeros(n, [('position', np.float32, 3),
                         ('color', np.float32, 4)])

        V['color'][:,3] = .1

        lim = np.max(sim['r'])
        norm = Normalize(-lim,lim)

        V['position'] = norm(sim['pos']) - 0.5

        if mode == 'rho' :
            ln = LogNorm()
            V['color'][:,0:3] = jet(ln(sim['rho']))[:,0:3]
            V['color'][:,3] = ln(sim['rho'])
        elif mode == 'type' : 
            V['color'] = [0,0,1,.5]

        
        fig = gp.figure((600,600))

#vshade = file('vshade_pixels').read()
#fshade = file('fshade_pixels').read()
        
#shader = gp.graphics.Shader(vshade, fshade)
#shader.bind()

        frame = fig.add_frame()
        mesh = gp.graphics.VertexBuffer( V )
        trackball = gp.Trackball( 0,0, .5, 5 )

        self.V = V
        self.fig = fig
        self.frame = frame
        self.mesh = mesh
        self.trackball = trackball
        
        self.init_handlers()

        gp.show()
    
    def init_handlers(self) : 
        @fig.event
        def on_mouse_press(x,y,button):
            global prev_mouse
            prev_mouse = 'press'
            
        @fig.event
        def on_mouse_release(x, y, button):
            global prev_mouse
            if prev_mouse == 'press' :
                if button == 2 : 
                    trackball.zoom_to(x,y,50,50)
                elif button == 8: 
                    trackball.zoom_to(x,y,-50,-50)
                fig.redraw()

        @fig.event
        def on_mouse_drag(x,y,dx,dy,button):
            global prev_mouse
            prev_mouse = 'drag'
            trackball.drag_to(x,y,dx,dy)
            fig.redraw()

        @fig.event
        def on_draw():
            fig.clear(0.85,0.85,0.85,1)
    
        @frame.event
        def on_draw():
            frame.lock()
            frame.draw()
            trackball.push()
            frame.clear(0,0,0,1)
#            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#            gl.glBlendEquation(gl.GL_MAX)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE)
            
            gl.glEnable(gl.GL_BLEND)

            gl.glEnable( gl.GL_POINT_SMOOTH )
            mesh.draw( gl.GL_POINTS, "pc" )
            gl.glDisable( gl.GL_POINT_SMOOTH )

            trackball.pop()
            frame.unlock()
    


    def update_alpha(self, alpha) : 
        V['color'][:,3] = alpha
        mesh.upload()
        fig.redraw()

    def update_point_size(self, point_size) : 
        gl.glPointSize(point_size)
        fig.redraw()
