from opengl_gui.gui_components import *
from gui_components import SettableRangeSlider, Colours, TouchContainer, Language, TextFieldMultilingual

class Command:
    DISABLE = 0
    ENABLE = 1

    CAMERA_STREAM_DEFAULT = 10
    CAMERA_STREAM_KINECT_AZURE = 11
    CAMERA_STREAM_AXIS_PTZ = 12

i8n_button_detect = {
    Language.EN: "Toggle detection",
    Language.SL: "Vključi detekcijo"
}
i8n_button_clear = {
    Language.EN: "Clear",
    Language.SL: "Počisti"
}
i8n_camera_default = {
    Language.EN: "Main camera",
    Language.SL: "Glavna kamera"
}
i8n_camera_kinect = {
    Language.EN: "Kinect Azure",
    Language.SL: "Kinect Azure"
}
i8n_camera_ptz = {
    Language.EN: "Axis PTZ",
    Language.SL: "Axis PTZ"
}
def get_scene(parameters):

    state = parameters.state
    echo = state.echolib_handler

    state.detection = 1
    state.counting_drag_start_pos = None
    state.counting_drag_current_pos = None
    state.counting_drag_stopped_time = None
    state.counting_exemplars = []
    state.counting_old_length = 0
    state.counting_threshold = 0.5
    state.counting_count = 0
    if not hasattr(state, "counting_camera_selected"):
        state.counting_camera_selected = Command.CAMERA_STREAM_DEFAULT

    button_scale = 1.4
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = Colours.VICOS_GRAY,
        on_click = toggle_detection,
        id       = "demo_count_button")

    button_text = TextFieldMultilingual(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect,
        id = "demo_count_text",
        language_callback = lambda field, lang: field.set_text(font = parameters.font, text = i8n_button_detect[lang]).center_x())

    button_text.set_text(font = parameters.font, text = i8n_button_detect[parameters.state.language])
    button_text.center_x()
    button_text.center_y()

    button_text.depends_on(element = button_detection)
    button_detection.center_x()
    
    button_exemplars = Button(
        position = [0.64, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = Colours.VICOS_RED,
        on_click = lambda button, gui, state: state.counting_exemplars.clear(),
        id       = "demo_exemplars_button")
    
    slider_threshold_text = TextField(
        position = [0.05, 0.0],
        text_scale = 0.68,
        colour = [1.0, 1.0, 1.0, 0.75],
        offset = [0, -0.05],
        aspect_ratio = parameters.aspect, 
        id = "slider_threshold_text")
    slider_threshold = SettableRangeSlider(
        position = [0.15, 0.95],
        scale = [0.15, 0.01],
        aspect_ratio = parameters.aspect,
        range_bottom=0,
        range_top=1,
        on_select = slider_threshold_on_select,
        id = "range_slider_threshold")
    slider_threshold_text.depends_on(slider_threshold)
    slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")
    
    if state.active_demo is not None:
        section = f"demos.{state.active_demo}"
        if section in state.config and "threshold" in state.config[section]:
            value = float(state.config[section]["threshold"])
            slider_threshold.set_value(value)
            slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")
                
            echo.append_command((echo.demo_counting_threshold, str(value)))
    slider_threshold.on_value_update  = lambda slider, state: \
        slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")
    
    container = TouchContainer(
        position = [0,0],
        scale = [1/parameters.state.get_aspect_ratio(),1],
        colour = [0,0,0,0],
        id = "counting_demo_touch_container",
        on_press = click_handler,
        on_move = move_handler
    )
    button_detection.depends_on(container)
    button_exemplars.depends_on(container)
    slider_threshold.depends_on(container)

    button_exemplars_text = TextFieldMultilingual(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect, 
        id = "demo_exemplars_text",
        language_callback = lambda field, lang: field.set_text(font = parameters.font, text = i8n_button_clear[lang]).center_x())

    button_exemplars_text.set_text(font = parameters.font, text = i8n_button_clear[parameters.state.language])
    button_exemplars_text.center_x()
    button_exemplars_text.center_y()

    button_exemplars_text.depends_on(element = button_exemplars)

    cam_selector_scale = button_scale
    cam_selector_pane = Container(
        position = [0.87, 0.877],
        scale    = [0.06*cam_selector_scale, 0.03*cam_selector_scale*2],
        colour   = [0,0,0,0],
        id       = "demo_cloth_cam_pane"
    )

    def get_switch_handler(camera_stream: int):
        def switch_camera(button: Button, gui: Gui, state):
            if state.echolib_handler.docker_channel_out is not None:
                state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, camera_stream))

            for b in cam_selector_pane.dependent_components:
                b.set_colour(Colours.VICOS_RED)

            state.counting_camera_selected = camera_stream
            button.set_colour(Colours.VICOS_GRAY)
        return switch_camera

    def add_camera_select_button(button_pane, id, stream_id, text):
        cam_selector = Button(
            position = [0.02, 0],
            scale    = [0.08*cam_selector_scale, 0.03*cam_selector_scale],
            offset   = [0, id*0.06*cam_selector_scale],
            colour   = Colours.VICOS_GRAY_NON_TRANSPARENT if state.counting_camera_selected == stream_id else Colours.VICOS_RED,
            on_click = get_switch_handler(stream_id),
            id       = "demo_counting_cam_{}".format(id))

        cam_selector_text = TextFieldMultilingual(
            colour   = [1.0, 1.0, 1.0, 1.0],
            position = [0.25, 0.5,],
            text_scale = 0.5,
            aspect_ratio = parameters.aspect,
            id = "demo_counting_cam_{}_text".format(id),
            language_callback = lambda field, lang: field.set_text(font = parameters.font, text = text[lang]).center_x())

        cam_selector_text.set_text(font = parameters.font, text = text[parameters.state.language])
        cam_selector_text.center_x()
        cam_selector_text.center_y()

        cam_selector_text.depends_on(element = cam_selector)
        cam_selector.center_x()
        cam_selector.depends_on(element = button_pane)

        return cam_selector

    add_camera_select_button(cam_selector_pane, 0, Command.CAMERA_STREAM_DEFAULT, i8n_camera_default)
    add_camera_select_button(cam_selector_pane, 1, Command.CAMERA_STREAM_KINECT_AZURE, i8n_camera_kinect)
    add_camera_select_button(cam_selector_pane, 2, Command.CAMERA_STREAM_AXIS_PTZ, i8n_camera_ptz)

    return { "get_docker_texture": get_docker_texture, "elements": [container, cam_selector_pane] }

def get_docker_texture(gui: Gui, state):
    echolib_handler = state.echolib_handler
    
    if not echolib_handler.docker_channel_ready:
        return None
    
    if state.demo_start:
        state.detection = 1
        echolib_handler.append_command((echolib_handler.docker_channel_out, Command.ENABLE))
        state.demo_start = False
    
    # return np.zeros((300, 400))
    image = echolib_handler.get_image()
    if image is None: return None
    
    height, width, _ = image.shape
    bboxes = state.counting_exemplars.copy()
    
    if state.counting_old_length != len(bboxes):
        if len(bboxes) > 0:
            bboxes_abs = np.array(bboxes) * np.array([width, height])
            bboxes_view = np.array(bboxes_abs.astype(np.float32).reshape((-1,4)))
        else:
            bboxes_view = np.array([[0,0,0,0]], dtype=np.float32)                    

        echolib_handler.append_command((echolib_handler.demo_counting_bboxes, bboxes_view))
        state.counting_old_length = len(bboxes)

    # Its important we don't send this one yet, as it wouldn't get re-sent
    if state.counting_drag_start_pos is not None:
        bboxes.append((state.counting_drag_start_pos, state.counting_drag_current_pos))
    for (x1,y1), (x2,y2) in bboxes:
        pt1 = int(x1 * width), int(y1 * height)
        pt2 = int(x2 * width), int(y2 * height)
        cv2.rectangle(image, pt1, pt2, (255, 255, 0), width//500)
    
    return image

def slider_threshold_on_select(slider: RangeSlider, state):
    if state.active_demo is not None:
        section = f"demos.{state.active_demo}"
        if section not in state.config:
            state.config[section] = {}
        state.config[section]["threshold"] = str(slider.selected_value)
    state.echolib_handler.append_command((state.echolib_handler.demo_counting_threshold, float(slider.selected_value)))


def toggle_detection(button: Button, gui: Gui, state):

    if state.echolib_handler.docker_channel_out is not None:

        toggle = (button.mouse_click_count + 1) % 2
        state.detection = toggle

        if toggle == 1:
            button.set_colour(colour = Colours.VICOS_GRAY)
        else:
            button.set_colour(colour = Colours.VICOS_RED)

        state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, Command.ENABLE if toggle == 1 else Command.DISABLE))


def click_handler(self, is_pressed, x, y, state):
    conv = self.to_local(x, y)
    # Outside of container or within dead zones for calibration menu
    if conv is None or conv[1] > 0.9:
        state.counting_drag_start_pos = None
        return
    x1,y1 = conv
    if is_pressed:
        state.counting_drag_start_pos = state.counting_drag_current_pos = (x1, y1)
    elif state.counting_drag_start_pos is not None:
        x2, y2 = state.counting_drag_start_pos
        # Ignore small drags
        if (x1 - x2)**2 + (y1 - y2)**2 > 0.001:
            # Make sure order is correct
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            state.counting_exemplars.append(((x1, y1), (x2, y2)))
        state.counting_drag_start_pos = None

def move_handler(self, x, y, state):
    if state.counting_drag_start_pos is not None:
        conv = self.to_local(x,y)
        if conv is None:
            state.counting_drag_start_pos = None
            return
        x,y = conv
        state.counting_drag_current_pos = (x, y)
