from opengl_gui.gui_components import *
from gui_components import SettableRangeSlider

import echolib
from echolib.camera import FramePublisher, Frame
from gui_components import Colours, TouchContainer

def get_scene(parameters):

    state = parameters.state
    state.detection = 1
    state.counting_publisher = FramePublisher(state.echolib_handler.client, "counting_bboxes")
    state.counting_publisher_threshold = echolib.Publisher(state.echolib_handler.client, "counting_threshold", "float")
    state.counting_drag_start_pos = None
    state.counting_drag_current_pos = None
    state.counting_drag_stopped_time = None
    state.counting_exemplars = []
    state.counting_old_length = 0
    state.counting_threshold = 0.5 # 0.4 - 0.6 is reasonable (1 == max detection on image)
    state.counting_count = 0
    state.counting_thresh_publish_buffer = None
    state.counting_thresh_publish_time = time.time()

    button_scale = 1.4
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = Colours.VICOS_GRAY,
        on_click = toggle_detection,
        id       = "demo_count_button")

    button_text = TextField(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect, 
        id = "demo_count_text")

    button_text.set_text(font = parameters.font, text = "Vključi detekcijo")
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
        id = "range_slider_threshold")
    slider_threshold_text.depends_on(slider_threshold)
    slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")
    
    if state.active_demo is not None:
        section = f"demos.{state.active_demo}"
        if section in state.config and "threshold" in state.config[section]:
            value = float(state.config[section]["threshold"])
            slider_threshold.set_value(value)
            slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")
                
            writer = echolib.MessageWriter()
            writer.writeFloat(value)
            state.counting_publisher_threshold.send(writer)
    slider_threshold.on_value_update  = lambda slider, state: \
        slider_threshold_text.set_text(font=parameters.font, text=f"Threshold: {slider_threshold.selected_value:.2f}")

    def slider_threshold_on_select(slider: RangeSlider, state):
        # Sending too many updates may cause echolib to die,
        # so we just save to state and debounce in get_docker_texture
        # Note: its still not 100% reliable
        state.counting_thresh_publish_buffer = slider.selected_value
        if state.active_demo is not None:
            section = f"demos.{state.active_demo}"
            if section not in state.config:
                state.config[section] = {}
            state.config[section]["threshold"] = str(slider.selected_value)
    slider_threshold.on_select = slider_threshold_on_select
    
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

    button_exemplars_text = TextField(
        colour   = [1.0, 1.0, 1.0, 1.0],
        position = [0.25, 0.65,],
        text_scale = 0.5,
        aspect_ratio = parameters.aspect, 
        id = "demo_exemplars_text")

    button_exemplars_text.set_text(font = parameters.font, text = "Počisti")
    button_exemplars_text.center_x()
    button_exemplars_text.center_y()

    button_exemplars_text.depends_on(element = button_exemplars)

    return { "get_docker_texture": get_docker_texture, "elements": [container] }

def get_docker_texture(gui: Gui, state):
    # print('docker texture', state.counting_count)
    # state.counting_count += 1
    if state.counting_thresh_publish_buffer is not None and time.time() - state.counting_thresh_publish_time > 3:
        print('sending', state.counting_thresh_publish_buffer)
        writer = echolib.MessageWriter()
        writer.writeFloat(state.counting_thresh_publish_buffer)
        state.counting_publisher_threshold.send(writer)
        state.counting_thresh_publish_buffer = None
        state.counting_thresh_publish_time = time.time()

    echolib_handler = state.echolib_handler

    if not echolib_handler.docker_channel_ready:
        return None
    
    if state.demo_start:
        state.detection = 1
        state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, 1))
        state.demo_start = False
    
    image = echolib_handler.get_image() if state.detection == 1 else echolib_handler.get_camera_stream()
    if image is None: return None
    # Watch out: if you're trying to increase responsiveness, image.copy() seems to cause some trouble
    
    height, width, _ = image.shape
    bboxes = state.counting_exemplars.copy()
    
    if state.counting_old_length != len(bboxes):
        if len(bboxes) > 0:
            bboxes_abs = np.array(bboxes) * np.array([width, height])
            bboxes_view = np.array(bboxes_abs
                                .astype(np.float32)
                                .reshape((-1,4))
                                .view(np.uint8), dtype=np.uint8)
        else:
            bboxes_view = np.array([[0,0,0,0]], dtype=np.uint8)                    

        state.counting_publisher.send(Frame(image=bboxes_view))
        state.counting_old_length = len(bboxes)

    # Its important we don't send this one yet, as it wouldn't get re-sent
    if state.counting_drag_start_pos is not None:
        bboxes.append((state.counting_drag_start_pos, state.counting_drag_current_pos))
    for (x1,y1), (x2,y2) in bboxes:
        pt1 = int(x1 * width), int(y1 * height)
        pt2 = int(x2 * width), int(y2 * height)
        cv2.rectangle(image, pt1, pt2, (255, 255, 0), 4)
    
    return image

def toggle_detection(button: Button, gui: Gui, state):

    if state.echolib_handler.docker_channel_out is not None:

        toggle = (button.mouse_click_count + 1) % 2
        state.detection = toggle

        if toggle == 1:
            button.set_colour(colour = Colours.VICOS_GRAY)
        else:
            button.set_colour(colour = Colours.VICOS_RED)

        state.echolib_handler.append_command((state.echolib_handler.docker_channel_out, toggle))


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
