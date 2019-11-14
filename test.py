import nibabel as nb
import numpy as np
import math
import nilearn

# ---- Topology ---- Load data using nibabel
gifti_image = nb.load("onlineAtlas/fs_LR.164k.L.flat.surf.gii")  # load left hemisphere
img_data = [x.data for x in gifti_image.darrays]
vertices_index = img_data[1].flatten()  # the indices of the flatmap (56588, 3)
vertices_index = np.array(vertices_index, dtype=np.uint32)
print(vertices_index.shape)

# ---- Underlay ---- Load nii. data (underlay color) and convert to grayscale value
vertices = img_data[0] / 250  # the coordinates of the flatmap vertices (28935, 3)
vertices_coord = np.array(vertices, dtype=np.float32)
nii_image = nb.load("onlineAtlas/fs_LR.164k.LR.sulc.dscalar.nii")
underlay_color = np.asarray(np.reshape(nii_image.get_data(), (nii_image.get_data().shape[1], )), dtype=np.float32)
underlay_color_L = np.split(underlay_color, 2)[0]  # the vertices color for left hemisphere
underlay_color_R = np.split(underlay_color, 2)[1]  # the vertices color for right hemisphere
min_left = underlay_color_L.min()  # Split into L and R hemispheres
max_left = underlay_color_L.max()
underlay_color_L = (underlay_color_L - min_left) / (max_left - min_left)
underlay_color_L = np.reshape(np.repeat(underlay_color_L, 3), (underlay_color_L.shape[0], 3))
underlay_render = np.concatenate((vertices_coord, underlay_color_L), axis=1)
underlay_render = np.concatenate((underlay_render, np.ones((underlay_render.shape[0], 1))), axis=1)
underlay_render = np.array(underlay_render.flatten(), dtype=np.float32)

# ---- Border ---- cerebellum and cortex
# Load border information (cerebellum)
# from numpy import genfromtxt
# border = genfromtxt('onlineAtlas/flatmap_border.csv', delimiter=',')
# bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0])
# border_info = np.concatenate((border, bcolor), axis=1).flatten()
# border_info = np.array(border_info / 100, dtype=np.float32)
# print(border_info.shape)

# (cortex)
border_img = nb.load("onlineAtlas/fs_LR.164k.L.border-IPS.func.gii")
border_data = [x.data for x in border_img.darrays][0]
border = vertices[np.where(border_data == 1)[0]]
bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0])  # set border default color
border_info = np.concatenate((border, bcolor), axis=1)
border_info = np.concatenate((border_info, np.ones((border_info.shape[0], 1))), axis=1)
border_info = np.array(border_info.flatten(), dtype=np.float32)


# ---- Overlay ---- Making overlay buffer object
alpha = 0.2  # set overlay transparency
from matplotlib import cm
overlay_img = nb.load("onlineAtlas/test.func.gii")  # load left hemisphere
overlay_data = [x.data for x in overlay_img.darrays][0]
overlay_color = cm.jet(overlay_data)  # covert to colormap
overlay_color[np.where(np.isnan(overlay_data)), 0:3] = underlay_color_L[np.where(np.isnan(overlay_data))]
overlay_color = overlay_color[:, 0:3]
overlay_render = np.concatenate((vertices_coord, overlay_color), axis=1)
overlay_render = np.concatenate((overlay_render, np.reshape(np.repeat(alpha, overlay_render.shape[0]), (overlay_render.shape[0], 1))), axis=1)
overlay_render = np.array(overlay_render.flatten(), dtype=np.float32)

# Python OpenGL entry
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy

vertex_shader = """
    #version 330
    in vec3 position;
    in vec4 color;

    out vec4 newColor;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
    }
    """

fragment_shader = """
    #version 330
    in vec4 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = newColor;
    }
    """


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


def main():

    # initialize glfw
    if not glfw.init():
        return

    w_width, w_height = 600, 600
    # glfw.window_hint(glfw.RESIZABLE, GL_FALSE)
    window = glfw.create_window(w_width, w_height, "Cortex Flatmap", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    glClear(GL_COLOR_BUFFER_BIT)
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Background color, default black

    glUseProgram(shader)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)


    # ----- the underlay rendering ----- #
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, underlay_render.shape[0] * 4, underlay_render, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertices_index.shape[0] * 4, vertices_index, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    glDrawElements(GL_TRIANGLES, vertices_index.shape[0], GL_UNSIGNED_INT, None)


    # ----- the overlay rendering ----- #
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, overlay_render.shape[0] * 4, overlay_render, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertices_index.shape[0] * 4, vertices_index, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    glDrawElements(GL_TRIANGLES, vertices_index.shape[0], GL_UNSIGNED_INT, None)


    # ----- border buffer object ----- #
    BBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, BBO)
    glBufferData(GL_ARRAY_BUFFER, border_info.shape[0] * 4, border_info, GL_STATIC_DRAW)

    b_position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(b_position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(b_position)

    b_color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(b_color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(b_color)
    glPointSize(3)
    glDrawArrays(GL_POINTS, 0, int(border_info.shape[0] / 7))

    glDisable(GL_BLEND)
    glDepthMask(GL_TRUE)

    glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
