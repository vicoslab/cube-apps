from opengl_gui.gui_components import *
from gui_components import Colours, Language, TextFieldMultilingual

import numpy as np

class Command:
    DISABLE = 0
    ENABLE = 1
    
    CAMERA_STREAM_DEFAULT = 10
    CAMERA_STREAM_KINECT_AZURE = 11
    CAMERA_STREAM_KINECT_V2 = 12

    CALIBRATE = 41
    GRAB = 42


i8n_button_detect = {
    Language.EN: "Toggle detection",
    Language.SL: "Vključi detekcijo"
}
i8n_button_calibrate = {
    Language.EN: "Calibrate",
    Language.SL: "Umeri"
}
i8n_button_grab = {
    Language.EN: "Grab cloth",
    Language.SL: "Zagrabi krpo"
}
i8n_camera_default = {
    Language.EN: "Main camera",
    Language.SL: "Glavna kamera"
}
i8n_camera_kinect = {
    Language.EN: "Kinect Azure",
    Language.SL: "Kinect Azure"
}

def get_scene(parameters):
        
    parameters.state.enable_detection = 1
    parameters.state.activate_camera = "default"

    def get_docker_texture(guci: Gui, state):

        echolib_handler = state.echolib_handler

        if not echolib_handler.docker_channel_ready:
            return None
        
        img = echolib_handler.get_image() #if state.enable_detection == 1 else echolib_handler.get_camera_stream()

        if img is not None:
            if state.demo_start:
                state.enable_detection = 1
                echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.ENABLE))
                state.demo_start = False

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

            toggle = (button.mouse_click_count + 1) % 2
            state.enable_detection = toggle

            if toggle == 1:
                button.set_colour(colour = Colours.VICOS_GRAY)
            else:
                button.set_colour(colour = Colours.VICOS_RED)

            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.ENABLE if toggle == 1 else Command.DISABLE))

    def grab_action(button: Button, gui: Gui, state):
        """ When Grab button is pressed sends out command for grabbing the object. """
        if state.echolib_handler.docker_channel_out is not None:
            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.GRAB))
        #print("Grab button pressed") # For debug

    def calibrate_action(button: Button, gui: Gui, state):
        """ When Grab button is pressed sends out command for mapping camera coordinates to robot coordiantes. """
        if state.echolib_handler.docker_channel_out is not None:
            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.CALIBRATE))
        #print("Calibrate button pressed") # For debug

    button_scale = 1.4

    # Button that enables grabing action
    button_grab = Button(
        position = [0.60, 0.92],
        scale    = [0.10 * button_scale, 0.03 * button_scale],
        colour   = Colours.VICOS_RED,
        on_click = grab_action,
        id       = "demo_cloth_text")

    grab_button_text = TextFieldMultilingual(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect,
        language_callback = lambda field, lang: field.set_text(font=parameters.font, text = i8n_button_grab[lang]).center_x(),
        id = "grab_demo_cloth_text")

    grab_button_text.set_text(font = parameters.font, text = i8n_button_grab[parameters.state.language])
    grab_button_text.center_x()
    grab_button_text.center_y()

    grab_button_text.depends_on(element = button_grab)
    #button_grab.center_x()
    
    # Button that enables calibration action
    button_calibrate = Button(
        position = [0.795, 0.92],
        scale    = [0.10 * button_scale, 0.03 * button_scale],
        colour   = Colours.VICOS_RED,
        on_click = calibrate_action,
        id       = "demo_calibrate_button")

    calibrate_button_text = TextFieldMultilingual(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect,
        language_callback = lambda field, lang: field.set_text(font = parameters.font, text = i8n_button_calibrate[lang]).center_x(),
        id = "calibrate_demo_cloth_text")
    
    calibrate_button_text.set_text(font = parameters.font, text = i8n_button_calibrate[parameters.state.language])
    calibrate_button_text.center_x()
    calibrate_button_text.center_y()

    calibrate_button_text.depends_on(element = button_calibrate)
    #button_calibrate.center_x()
    
    # Button that enables detection
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = Colours.VICOS_GRAY,
        on_click = toggle_detection,
        id       = "demo_cloth_button")

    button_text = TextFieldMultilingual(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect,
        language_callback = lambda field, lang: field.set_text(font = parameters.font, text = i8n_button_detect[lang]).center_x(),
        id = "demo_cloth_text")

    button_text.set_text(font = parameters.font, text = i8n_button_detect[parameters.state.language])
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
            b.set_colour(Colours.VICOS_RED)
        
        button.set_colour(Colours.VICOS_GRAY)
    
    def add_camera_select_button(button_pane, id, callback, text, position, enabled=True):
        cam_selector = Button(
            position = [0.02, position],
            scale    = [0.08*cam_selector_scale, 0.03*cam_selector_scale],
            colour   = Colours.VICOS_RED if enabled else Colours.VICOS_GRAY,
            on_click = callback,
            id       = "demo_cloth_cam_{}".format(id))

        cam_selector_text = TextFieldMultilingual(
            colour   = [1.0, 1.0, 1.0, 1.0],
            position = [0.25, 0.5,],
            text_scale = 0.4,
            aspect_ratio = parameters.aspect,
            language_callback = lambda field, lang: field.set_text(font = parameters.font, text = text[lang]).center_x(),
            id = "demo_cloth_cam_{}_text".format(id))

        cam_selector_text.set_text(font = parameters.font, text = text[parameters.state.language])
        cam_selector_text.center_x()
        cam_selector_text.center_y()

        cam_selector_text.depends_on(element = cam_selector)
        cam_selector.center_x()
        cam_selector.depends_on(element = button_pane)

        return cam_selector

    from functools import partial
    add_camera_select_button(cam_selector_pane, 1, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_DEFAULT), i8n_camera_default, position=0.01, enabled=True)
    add_camera_select_button(cam_selector_pane, 2, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_KINECT_AZURE), i8n_camera_kinect, position=0.34, enabled=False)
    #add_camera_select_button(cam_selector_pane, 3, partial(switch_camera,camera_stream=Command.CAMERA_STREAM_KINECT_V2), "Kinect v2", position=0.67, enabled=True)
    
    return {"get_docker_texture": get_docker_texture, "elements": [button_detection, cam_selector_pane, button_grab, button_calibrate,]}
