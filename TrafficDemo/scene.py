from opengl_gui.gui_components import *

def get_scene(parameters):
    
    parameters.state.traffic_detection = 0

    vicos_gray = [85.0/255.0, 85.0/255.0, 85.0/255.0, 0.75]
    vicos_red  = [226.0/255, 61.0/255, 40.0/255.0, 0.75]

    def get_docker_texture(gui: Gui, state):

        echolib_handler = state.echolib_handler

        if not echolib_handler.docker_channel_ready:
            return None
        
        return echolib_handler.get_image() if state.traffic_detection == 1 else echolib_handler.get_camera_stream()

    def toggle_detection(button: Button, gui: Gui, state):

        if state.echolib_handler.docker_channel_out is not None:

            toggle = button.mouse_click_count % 2
            state.traffic_detection = toggle

            if toggle == 1:
                button.set_colour(colour = vicos_gray)
            else:
                button.set_colour(colour = vicos_red)

            state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, toggle))


    button_scale = 1.4
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = vicos_red,
        on_click = toggle_detection,
        id       = "demo_traffic_toggle_button")

    button_text = TextField(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect, 
        id = "demo_traffic_text")

    button_text.set_text(font = parameters.font, text = "Vključi detekcijo")
    button_text.center_x()
    button_text.center_y()

    button_text.depends_on(element = button_detection)
    button_detection.center_x()

    return {"get_docker_texture": get_docker_texture, "elements": [button_detection]}