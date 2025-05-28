from opengl_gui.gui_components import *
from echolib.camera import FramePublisher, Frame
from gui_components import Colours, TouchContainer

def get_scene(parameters):

    parameters.state.detection = 0
    parameters.state.counting_publisher = FramePublisher(parameters.state.echolib_handler.client, "counting_bboxes")
    parameters.state.counting_drag_start_pos = None
    parameters.state.counting_drag_current_pos = None
    parameters.state.counting_drag_stopped_time = None
    parameters.state.counting_exemplars = []
    parameters.state.counting_old_length = 0  

    button_scale = 1.4
    button_detection = Button(
        position = [0.44, 0.92],
        scale    = [0.10*button_scale, 0.03*button_scale],
        colour   = Colours.VICOS_RED,
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

    echolib_handler = state.echolib_handler

    if not echolib_handler.docker_channel_ready:
        return None
    
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
                                .reshape((1,-1,4))
                                .view(np.uint8), dtype=np.uint8)
        else:
            bboxes_view = np.array([[[0,0,0,0]]], dtype=np.uint8)                    

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

        toggle = button.mouse_click_count % 2
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
