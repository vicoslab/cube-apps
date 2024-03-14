from opengl_gui.gui_components import *

import numpy as np

class Command:
    DISABLE = 0
    ENABLE = 1
    
    CAMERA_STREAM_DEFAULT = 10
    CAMERA_STREAM_KINECT_AZURE = 11
    CAMERA_STREAM_KINECT_V2 = 12

def get_scene(parameters):
        
    parameters.state.enable_detection = 0
    parameters.state.activate_camera = "default"

    vicos_gray = [85.0/255.0, 85.0/255.0, 85.0/255.0, 0.75]
    vicos_red  = [226.0/255, 61.0/255, 40.0/255.0, 0.75]

    def get_docker_texture(guci: Gui, state):

        echolib_handler = state.echolib_handler

        if not echolib_handler.docker_channel_ready:
            return None
        
        img = echolib_handler.get_image() #if state.enable_detection == 1 else echolib_handler.get_camera_stream()

        if img is not None:
            expected_aspect_ratio = state.get_aspect_ratio()
            img_aspect_ratio = img.shape[1] / img.shape[0]

            # pad image if aspect ratio is not the same
            #if expected_aspect_ratio > img_aspect_ratio:
            #        pad = np.abs(int((img.shape[0] * expected_aspect_ratio - img.shape[1]) / 2))
            #        img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=0)
            #elif expected_aspect_ratio < img_aspect_ratio:
            #        pad = np.abs(int((img.shape[1] / expected_aspect_ratio - img.shape[0]) / 2))
            #        img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
            if expected_aspect_ratio > img_aspect_ratio:
                    crop = np.abs(int((img.shape[1] / expected_aspect_ratio - img.shape[0]) / 2))
                    img = img[crop:-crop,:]
            elif expected_aspect_ratio < img_aspect_ratio:
                    crop = np.abs(int((img.shape[0] * expected_aspect_ratio - img.shape[1]) / 2))
                    img = img[:,crop:-crop]
                    
        return img

    def toggle_detection(button: Button, gui: Gui, state):

        if state.echolib_handler.docker_channel_out is not None:

            toggle = button.mouse_click_count % 2
            state.enable_detection = toggle

            if toggle == 1:
                button.set_colour(colour = vicos_gray)
            else:
                button.set_colour(colour = vicos_red)

            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.ENABLE if toggle == 1 else Command.DISABLE))


    button_scale = 1.4
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = vicos_red,
        on_click = toggle_detection,
        id       = "demo_cloth_button")

    button_text = TextField(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect, 
        id = "demo_cloth_text")

    button_text.set_text(font = parameters.font, text = "Vključi detekcijo")
    button_text.center_x()
    button_text.center_y()

    button_text.depends_on(element = button_detection)
    button_detection.center_x()

    cam_selector_scale = 1.2
    cam_selector_pane = Container(
        position = [0.02, 0.89],
        scale    = [0.06*cam_selector_scale, 0.03*cam_selector_scale*3.05],
        colour   = [1.0, 1.0, 1.0, 0.1],
        id       = "demo_cloth_cam_pane"
    )

    def switch_camera(button: Button, gui: Gui, state, camera_stream: int):
        if state.echolib_handler.docker_channel_out is not None:
            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, camera_stream))
        
        for b in cam_selector_pane.dependent_components:
            b.set_colour(vicos_red)
        
        button.set_colour(vicos_gray)
    
    def add_camera_select_button(button_pane, id, callback, text, position, enabled=True):
        cam_selector = Button(
            position = [0.02, position],
            scale    = [0.08*cam_selector_scale, 0.03*cam_selector_scale],
            colour   = vicos_red if enabled else vicos_gray,
            on_click = callback,
            id       = "demo_cloth_cam_{}".format(id))

        cam_selector_text = TextField(
            colour   = [1.0, 1.0, 1.0, 1.0],
            position = [0.25, 0.5,],
            text_scale = 0.4,
            aspect_ratio = parameters.aspect, 
            id = "demo_cloth_cam_{}_text".format(id))

        cam_selector_text.set_text(font = parameters.font, text = text)
        cam_selector_text.center_x()
        cam_selector_text.center_y()

        cam_selector_text.depends_on(element = cam_selector)
        cam_selector.center_x()
        cam_selector.depends_on(element = button_pane)

        return cam_selector

    from functools import partial
    add_camera_select_button(cam_selector_pane, 1, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_DEFAULT), "Glavna kamera", position=0.01, enabled=False)
    add_camera_select_button(cam_selector_pane, 2, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_KINECT_AZURE), "Kinect Azure", position=0.34, enabled=True)
    #add_camera_select_button(cam_selector_pane, 3, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_KINECT_V2), "Kinect v2", position=0.67, enabled=True)

    return {"get_docker_texture": get_docker_texture, "elements": [button_detection, cam_selector_pane]}
