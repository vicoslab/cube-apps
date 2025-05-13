from opengl_gui.gui_components import *
from echolib.camera import FramePublisher, Frame

def get_scene(parameters):
    
    parameters.state.detection = 0
    parameters.state.counting_publisher = FramePublisher(parameters.state.echolib_handler.client, "counting_bboxes")
    parameters.state.counting_drag_start_pos = None
    parameters.state.counting_drag_current_pos = None
    parameters.state.counting_drag_stopped_time = None
    parameters.state.counting_exemplars = []
    parameters.state.counting_old_length = 0

    vicos_gray = [85.0/255.0, 85.0/255.0, 85.0/255.0, 0.75]
    vicos_red  = [226.0/255, 61.0/255, 40.0/255.0, 0.75]

    def get_docker_texture(gui: Gui, state):

        echolib_handler = state.echolib_handler

        if not echolib_handler.docker_channel_ready:
            return None
        
        image = echolib_handler.get_image() if state.detection == 1 else echolib_handler.get_camera_stream()
        if image is not None:
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

            # Its important we don't send this one yet, as it is not finished
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
        colour   = vicos_red,
        on_click = lambda button, gui, state: state.counting_exemplars.clear(),
        id       = "demo_exemplars_button")
        
    def element_update(parent, gui: Gui, custom_data):
        button_exemplars.element_update(parent, gui, custom_data)

        left, top = parent.top
        right, bottom = parent.bot
        
        state = custom_data
        if gui.mouse_press_event is not None and gui.interaction_context_free(button_exemplars):

            is_pressed, x, y = gui.mouse_press_event
            if x < left or x > right or y < bottom or y > top:
                state.counting_drag_start_pos = None
                return
            x = (x - left) / (right - left)
            y = 1 - (y - bottom) / (top - bottom)
            
            if is_pressed:
                state.counting_drag_start_pos = state.counting_drag_current_pos = (x, y)
            else:
                x1, y1 = state.counting_drag_start_pos
                # Ignore small drags
                if (x1 - x)**2 + (y1 - y)**2 > 0.001:
                    state.counting_exemplars.append((state.counting_drag_start_pos, (x, y)))
                state.counting_drag_start_pos = None

        if state.counting_drag_start_pos is not None:
            if gui.x_pos < left or gui.x_pos > right or gui.y_pos < bottom or gui.y_pos > top:
                state.counting_drag_start_pos = None
                return
            x = (gui.x_pos - left) / (right - left)
            y = 1 - (gui.y_pos - bottom) / (top - bottom)
            
            state.counting_drag_current_pos = (x, y)
        #print("exemplars", state.counting_exemplars)
    # Hook into button_exemplars' update function
    button_exemplars.command_chain = [
        element_update,
        button_exemplars.element_render,
        button_exemplars.element_exit
    ]

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

    return { "get_docker_texture": get_docker_texture, "elements": [button_detection, button_exemplars] }