import nibabel as nb
import numpy as np
from matplotlib import cm
# Python OpenGL entry
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

"""
    Initialize vertex shader and fragment shader program
"""
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


def load_topo(filename):
    """
    # ---- Topology ---- Load data using nibabel
    Input:
        filename: the filename with path. (eg. "onlineAtlas/fs_LR.164k.L.flat.surf.gii")
    Returns: the topology of the flatmap
        vertices_coord: the coordinates of the vertices, shape: (N, 3)
        vertices_index: the vertices connectivity info used for indexed drawing. shape: flatten
    """
    gifti_image = nb.load(filename)  # load left hemisphere
    img_data = [x.data for x in gifti_image.darrays]
    vertices = img_data[0] / 250  # the coordinates of the flatmap vertices (N, 3)
    vertices_coord = np.array(vertices, dtype=np.float32)
    vertices_index = img_data[1].flatten()  # the indices of the flatmap (N, 3)
    vertices_index = np.array(vertices_index, dtype=np.uint32)
    print("Topology loaded: node size: " + str(vertices_coord.shape[0]) + " faces:" + str(vertices_index.shape[0]/3))

    return vertices_coord, vertices_index


def load_underlay_buffer(filename, vertices_c, hemisphere='left'):
    """
    # ---- Making Underlay buffer----
        Load vertices coordinates and combine with underlay color (converted to grayscale value)
        to make underlay buffer data
    Input:
        filename: the underlay file name with path. (eg. "onlineAtlas/fs_LR.164k.LR.sulc.dscalar.nii")
        vertices_coord: the vertices coordinates of the flatmap. shape (N, 3)
        hemisphere: 'left' or 'right', determine which hemisphere want to render, default: left
    Returns: the flatten buffer data for underlay
        underlay_render: the underlay buffer data for rendering. shape: flatten
        underlay_color: the underlay color itself, shape (N, 3)
    """
    nii_image = nb.load(filename)
    underlay_color = np.asarray(np.reshape(nii_image.get_data(), (nii_image.get_data().shape[1], )), dtype=np.float32)

    if hemisphere is 'left':
        underlay_color = np.split(underlay_color, 2)[0]
    elif hemisphere is 'right':
        underlay_color = np.split(underlay_color, 2)[1]
    else:
        raise TypeError("hemisphere must be 'left' or 'right'!")

    min = underlay_color.min()
    max = underlay_color.max()
    underlay_color = (underlay_color - min) / (max - min)
    underlay_color = np.reshape(np.repeat(underlay_color, 3), (underlay_color.shape[0], 3))
    underlay_render = np.concatenate((vertices_c, underlay_color), axis=1)
    underlay_render = np.concatenate((underlay_render, np.ones((underlay_render.shape[0], 1))), axis=1)  # Concatenate with alpha value (default = 1)
    underlay_render = np.array(underlay_render.flatten(), dtype=np.float32)

    return underlay_render, underlay_color


def load_borders_buffer(filenames, vertices_coord, flatmap_type='cortex'):
    """
    # ---- Making borders buffer ----
        Look up the vertices coord to find which node is the border and make it black color for rendering
    Input:
        filenames: the border files name with path. (eg. "onlineAtlas/fs_LR.164k.L.border-IPS.func.gii")
        vertices_coord: the vertices coordinates of the flatmap. shape (N, 3)
        flatmap_type: the type of flatmap. ('cerebellum' or 'cortex')
    Returns: the borders buffer data for rendering
        borders: the borders buffer data for rendering. shape: list(borders#, )
    """
    borders = []
    if flatmap_type is 'cortex':
        for file in filenames:  # (cortex)
            border_img = nb.load(file)
            border_data = [x.data for x in border_img.darrays][0]
            border = vertices_coord[np.where(border_data == 1)[0]]
            bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0])  # set border default color (black)
            border_info = np.concatenate((border, bcolor), axis=1)
            border_info = np.concatenate((border_info, np.ones((border_info.shape[0], 1))), axis=1)  # set alpha = 1.0
            border_info = np.array(border_info.flatten(), dtype=np.float32)
            borders.append(border_info)
    elif flatmap_type is 'cerebellum':
        # ---- Border ---- cerebellum and cortex
        # Load border information (cerebellum)
        from numpy import genfromtxt
        border = genfromtxt('onlineAtlas/flatmap_border.csv', delimiter=',')
        bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0])
        border_info = np.concatenate((border, bcolor), axis=1).flatten()
        border_info = np.array(border_info / 100, dtype=np.float32)
        borders = border_info
    else:
        raise TypeError("flatmap type must be 'cortex' or 'cerebellum'!")

    return borders


def load_overlays_buffer(filenames, vertices_coord, underlay_color, cmap='jet', alpha=0.2, overlay_type='cortex'):
    """
    # ---- Making Underlay buffer----
        Load vertices coordinates and combine with overlay color (converted to selected cmap value)
        to make overlays buffer data for rendering
        Note: Currently, this function only support single condition contrast
    Input:
        filenames: the overlay files name with path. (eg. "onlineAtlas/data.func.gii")
        vertices_coord: the vertices coordinates of the flatmap. shape (N, 3) from load_topo function
        cmap: see 'matplotlib' colormaps for more selections
        alpha: the transparency value for these overlays, default 0.2
        overlay_type: the overlay type, 'cortex' or 'cerebellum'
    Returns: the flatten buffer data for underlay
        overlays: the overlays buffer data for rendering. shape: list(overlays#, )
    """
    overlays = []
    if overlay_type is 'cortex':
        for file in filenames:
            overlay_img = nb.load(file)  # load left hemisphere
            overlay_data = [x.data for x in overlay_img.darrays][0]
            overlay_color = cm.jet(overlay_data)  # covert to colormap
            overlay_color[np.where(np.isnan(overlay_data)), 0:3] = underlay_color[np.where(np.isnan(overlay_data))]
            overlay_color = overlay_color[:, 0:3]
            overlay_render = np.concatenate((vertices_coord, overlay_color), axis=1)
            overlay_render = np.concatenate((overlay_render, np.reshape(np.repeat(alpha, overlay_render.shape[0]), (overlay_render.shape[0], 1))), axis=1)
            overlay_render = np.array(overlay_render.flatten(), dtype=np.float32)
            overlays.append(overlay_render)
    elif overlay_type is 'cerebellum':
        raise TypeError("I'm sorry, currently not support cerebellum overlays ...")
    else:
        raise TypeError("Overlay type must be 'cortex' or 'cerebellum'!")

    print(str(len(overlays)) + " overlays have been loaded ...")
    return overlays


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


def render_underlay(underlay, vertices_index, shader):
    """
    # ---- Rendering underlay contrast ----
    Input:
        underlay: the vertices coordinates buffer of the flatmap for the underlay
        vertices_index: the flatmap vertices drawing order (faces)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing underlay to the scene
    """
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, underlay.shape[0] * 4, underlay, GL_STATIC_DRAW)

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


def render_overlays(overlays_buffer, vertices_index, shader):
    """
    # ---- Rendering overlays contrast ----
    Input:
        overlays_buffer: the vertices coordinates buffers (flatten)
        vertices_index: the flatmap vertices drawing order (faces)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing overlay to the scene
    """
    for overlay in overlays_buffer:
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, overlay.shape[0] * 4, overlay, GL_STATIC_DRAW)

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


def render_borders(borders_buffer, shader):
    """
    # ---- Rendering borders ----
    Input:
        borders_buffer: the border buffers objects (flatten)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing selected borders to the scene
    """
    for border_info in borders_buffer:
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


def render(vertices_index, borders, underlay, overlays):
    """
    # ---- The main entry of the OpenGL rendering ----
    Input:
        vertices_index: the flatmap vertices drawing order (faces)
        borders: the border buffers objects (flatten) shape list(N, )
        underlay: the underlay buffers object
        overlays: the overlays buffer object (flatten), shape list(N, )
    Returns:
        Draw the scene
    """
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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Background color, default black

    glUseProgram(shader)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)

    # rendering objects
    render_underlay(underlay, vertices_index, shader)  # -- the underlay rendering
    render_overlays(overlays, vertices_index, shader)  # -- the overlays rendering
    render_borders(borders, shader)  # -- border buffer object

    glDisable(GL_BLEND)  # Disable gl blending from this point
    glDepthMask(GL_TRUE)

    glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    print('Start rendering cortical flatmap ...')

    topo = "onlineAtlas/fs_LR.164k.L.flat.surf.gii"
    underlay = "onlineAtlas/fs_LR.164k.LR.sulc.dscalar.nii"

    overlays = ["onlineAtlas/test.func.gii",
                "onlineAtlas/test2.func.gii"]

    borders = ["onlineAtlas/fs_LR.164k.L.border-CS.func.gii",
               "onlineAtlas/fs_LR.164k.L.border-IPS.func.gii",
               "onlineAtlas/fs_LR.164k.L.border-PoCS.func.gii",
               "onlineAtlas/fs_LR.164k.L.border-SF.func.gii"]

    vertices_coord, vertices_index = load_topo(topo)
    underlay_buffer, underlay_color = load_underlay_buffer(underlay, vertices_coord, hemisphere='left')
    overlays_buffer = load_overlays_buffer(overlays, vertices_coord, underlay_color)
    borders_buffer = load_borders_buffer(borders, vertices_coord)

    render(vertices_index, borders_buffer, underlay_buffer, overlays_buffer)

