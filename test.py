import nibabel as nb
import numpy as np
import nilearn

# Load data using nibabel
gifti_image = nb.load("onlineAtlas/fs_LR.164k.L.flat.surf.gii")
img_data = [x.data for x in gifti_image.darrays]
vertices_coord = img_data[0].flatten() / 250  # the coordinates of the flatmap vertices (28935, 3)
vertices_coord = np.array(vertices_coord, dtype=np.float32)
vertices_index = img_data[1].flatten()  # the indices of the flatmap (56588, 3)
vertices_index = np.array(vertices_index, dtype=np.uint32)
print(vertices_coord.shape, vertices_index.shape)

# Load nii. data (underlay color)
nii_image = nb.load("onlineAtlas/fs_LR.164k.LR.sulc.dscalar.nii")
underlay_color = nii_image.get_data()

# Load border information
from numpy import genfromtxt
border = genfromtxt('onlineAtlas/flatmap_border.csv', delimiter=',')
bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0])
border_info = np.concatenate((border, bcolor), axis=1).flatten()
print(border_info.shape)

# Load contrast if available, otherwise the default flatmap color is grey (0.8, 0.8, 0.8)


# # nilearn part - load nii. img
# img = nilearn.image.load_img('onlineAtlas/Cond01_No-Go.nii')
# # img = nilearn.image.smooth_img('onlineAtlas/Cond01_No-Go.nii') # load and smooth data

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy


def main():

    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(600, 600, "Cerebellum Flatmap", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    vertex_shader = """
    #version 330
    in vec3 position;
    in vec3 color;

    out vec3 newColor;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = vec4(0.8f, 0.8f, 0.8f, 1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices_coord.shape[0] * 4, vertices_coord, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertices_index.shape[0] * 4, vertices_index, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    # color = glGetAttribLocation(shader, "color")
    # glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(color)

    # border buffer object
    BBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, BBO)
    glBufferData(GL_ARRAY_BUFFER, border_info.shape[0] * 4, border_info, GL_STATIC_DRAW)

    # b_position = glGetAttribLocation(shader, "position")
    # glVertexAttribPointer(b_position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    # glEnableVertexAttribArray(b_position)
    #
    # b_color = glGetAttribLocation(shader, "color")
    # glVertexAttribPointer(b_color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(b_color)

    glUseProgram(shader)

    glClearColor(0.0, 0.0, 0.0, 1.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)

        glDrawElements(GL_TRIANGLES, vertices_index.shape[0], GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
