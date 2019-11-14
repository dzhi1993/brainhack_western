import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import array
from OpenGL.arrays import vbo
import numpy as np


class Scene:
    """ OpenGL 2D scene class """

    # initialization
    def __init__(self, width, height):
        self.numberofpoints = 1000
        self.pointlist = []
        self.pointsize = 10
        self.linesize = 3
        self.width = width
        self.height = height
        self.bezierpoints = []
        self.deboorpoints = []
        self.ordnung = 3
        glPointSize(self.pointsize)
        glLineWidth(self.linesize)

    # render
    def render(self):
        # render a point
        if len(self.pointlist) > 0:
            glClear(GL_COLOR_BUFFER_BIT)
            glLoadIdentity()
            glColor3f(0,0,0)
            self.drawPoints()
            self.drawPolygon()
            if self.ordnung < len(self.pointlist):
                self.drawBSplineCurve()

    def drawPoints(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        self.vbo = vbo.VBO(array(self.pointlist, 'f'))
        self.vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_POINTS, 0, len(self.pointlist))
        self.vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def drawPolygon(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        self.vbo = vbo.VBO(array(self.pointlist, 'f'))
        self.vbo.bind()
        glColor3f(0, 0, 0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_LINE_STRIP, 0, len(self.pointlist))
        self.vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def drawBezierCurve(self):
        t = 0.0
        if len(self.pointlist) > 1:
            while (t <= 1):
                self.casteljau(t)
                t += self.numberofpoints

        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        self.vbo = vbo.VBO(array(self.bezierpoints, 'f'))
        self.vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_POINTS, 0, len(self.bezierpoints))
        self.vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        del self.bezierpoints[:]

    def drawBSplineCurve(self):
        """draws a splinecurve from given controlpoints"""
        controlpoints = self.pointlist
        knotvector = []
        knotvector.extend([0 for x in range(self.ordnung)])
        last_entry = len(self.pointlist) - (self.ordnung - 2)
        knotvector.extend([x for x in range(1, last_entry)])
        knotvector.extend([last_entry for x in range(self.ordnung)])
        for i in np.arange(knotvector[0], knotvector[-1], (knotvector[-1] - knotvector[0]) / float(self.numberofpoints)):
            # starts deboor
            self.deboor(self.ordnung, controlpoints, knotvector, i)

        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        self.vbo = vbo.VBO(array(self.deboorpoints, 'f'))
        self.vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, self.vbo)
        glDrawArrays(GL_POINTS, 0, len(self.deboorpoints))
        self.vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        del self.deboorpoints[:]

    def deboor(self,degree,controlpoints,knotvector,t):
        """deboor algorithm"""
        r = 0
        for i in range(len(knotvector)):
            if knotvector[i] > t:
                r = i - 1
                break
        j = 1
        i = r
        curvepoints = []
        curvepoints.append(self.pointlist)
        for x in range(1, degree + 1):
            curvepoints.append([])

        for j in range(1, degree):
            for i in range(r, r - degree + 2 + j):
                alpha = (t - knotvector[i]) / (knotvector[i - j + degree] - knotvector[i])
                x = (1 - alpha) * curvepoints[j - 1][i - 1][0] + alpha * curvepoints[j - 1][i][0]
                y = x = (1 - alpha) * curvepoints[j - 1][i - 1][1] + alpha * curvepoints[j - 1][i][1]
                point = [x, y]
                curvepoints[j][i] = point

        self.deboorpoints.append(point)





    def casteljau(self,t, a=0, b=1):
        i = 0
        r = 1
        n = len(self.pointlist) - 1
        curvepoints = []
        curvepoints.append(self.pointlist)
        for x in range(1, n + 1):
            curvepoints.append([])
        while (r <= n):
            i = 0
            while (i <= n - r):
                p1 = curvepoints[r - 1][i]
                p2 = curvepoints[r - 1][i + 1]
                p = (float((b - t)) / (b - a) * p1[0] + float((t - a)) / (b - a) * p2[0]), (
                            float((b - t)) / (b - a) * p1[1] + float((t - a)) / (b - a) * p2[1])
                curvepoints[r].append(p)
                i += 1

            r += 1
        self.bezierpoints.append(p)

class RenderWindow:
    """GLFW Rendering window class"""

    def __init__(self):

        # save current working directory
        cwd = os.getcwd()

        # Initialize the library
        if not glfw.init():
            return

        # restore cwd
        os.chdir(cwd)

        # version hints
        # glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        # glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # buffer hints
        glfw.window_hint(glfw.DEPTH_BITS, 32)

        # define desired frame rate
        self.frame_rate = 100

        # make a window
        self.width, self.height = 400, 400
        self.aspect = self.width / float(self.height)
        self.window = glfw.create_window(self.width, self.height, "2D Graphics", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)

        # initialize GL
        glViewport(0, 0, self.width, self.height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glOrtho(0, self.width, 0, self.height, 1, -1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # set window callbacks
        glfw.set_mouse_button_callback(self.window, self.onMouseButton)
        glfw.set_key_callback(self.window, self.onKeyboard)

        # create 3D
        self.scene = Scene(self.width, self.height)

        # exit flag
        self.exitNow = False


    def onMouseButton(self, win, button, action, mods):
        if action is glfw.PRESS:

            if button == glfw.MOUSE_BUTTON_LEFT:
                point = glfw.get_cursor_pos(win)
                self.scene.pointlist.append([point[0],self.height-point[1]])


    def onKeyboard(self, win, key, scancode, action, mods):
        print("keyboard: ", win, key, scancode, action, mods)
        if action == glfw.PRESS:
            # ESC to quit
            if key == glfw.KEY_ESCAPE:
                self.exitNow = True
            if key == glfw.KEY_V:
                # toggle show vector
                self.scene.showVector = not self.scene.showVector

            if key == glfw.KEY_N:
                self.scene.numberofpoints += 0.001
            if key == glfw.KEY_C:
                # toggle animation
                del self.scene.bezierpoints[:]
                del self.scene.pointlist[:]


    def run(self):
        # initializer timer
        glfw.set_time(0.0)
        t = 0.0
        while not glfw.window_should_close(self.window) and not self.exitNow:
            # update every x seconds
            currT = glfw.get_time()
            if currT - t > 1.0 / self.frame_rate:
                # update time
                t = currT
                # clear
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # render scene
                self.scene.render()

                glfw.swap_buffers(self.window)
                # Poll for and process events
                glfw.poll_events()
        # end
        glfw.terminate()


# main() function
def main():
    print("Simple glfw render Window")
    rw = RenderWindow()
    rw.run()


# call main
if __name__ == '__main__':
    main()