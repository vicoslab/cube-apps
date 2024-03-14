#!/usr/bin/python
import torch
import numpy as np
import cv2

import model_args
from models import get_model, get_center_model
from torchvision import transforms

import argparse
from echolib_wrapper import EcholibWrapper

class ClothDemo:
    def __init__(self, MODEL_PATH, CENTER_MODEL_PATH=None):
        args = model_args.get_args()

        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

        self.model = get_model(args['model']['name'], args['model']['kwargs'])
        self.model.init_output(args['num_vector_fields'])
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                        is_learnable=args['center_model'].get('use_learnable_center_estimation'),
                                        use_fast_estimator=True)
        self.center_model.init_output(args['num_vector_fields'])
        self.center_model = torch.nn.DataParallel(self.center_model).to(self.device)

        state = torch.load(MODEL_PATH)
        self.model.load_state_dict(state['model_state_dict'], strict=True)

        # load center model from another file if specified so
        if CENTER_MODEL_PATH is not None:
            state = torch.load(CENTER_MODEL_PATH)

        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
        state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]
        self.center_model.load_state_dict(state['center_model_state_dict'], strict=False)

        self.model.eval()
        self.center_model.eval()

        self.to_tensor = transforms.ToTensor()

        self.threshold = args["threshold"]
        self.padding = args["padding"] if "padding" in args else 0
        self.size = args["size"]
        
        self.resize = transforms.Resize((self.size - 2*self.padding, self.size - 2*self.padding))
        self.pad = transforms.Pad((self.padding, self.padding))
        
        
        

    def predict(self, image):
        image_tensor = self.to_tensor(image)
        image_tensor = self.resize(image_tensor)

        model_input = self.pad(image_tensor)

        # cv2.imshow("padded", model_input.permute(1,2,0).cpu().numpy())
        # cv2.waitKey(1)

        with torch.inference_mode():
            output_batch_ = self.model(model_input)
            center_output = self.center_model(output_batch_)

        center_pred = center_output["center_pred"]
        pred_angle = center_output["pred_angle"]

        predictions = center_pred[0][center_pred[0,:,0] == 1][:,1:].cpu().numpy()
        idx = np.argsort(predictions[:, -1])
        idx = idx[::-1]
        predictions = predictions[idx, :]
        angles = pred_angle[0].cpu().numpy()[idx, :]

        for prediction, angle in zip(predictions, angles):
            if prediction[3] < self.threshold:
                continue

            scale_y = image.shape[0] / (self.size - 2*self.padding)
            scale_x = image.shape[1] / (self.size - 2*self.padding)

            x1, y1 = int((prediction[0] - self.padding / 2) * scale_x) * 2, int((prediction[1] - self.padding / 2) * scale_y) * 2

            if x1 < 0 or image.shape[1] < x1:
                continue

            if y1 < 0 or image.shape[0] < y1:
                continue

            angle = (angle - 180) * np.pi / 180
            length = image.shape[1] / 30
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))

            image = cv2.line(image, (x1, y1), (x2, y2), color=(255,0,0), thickness=image.shape[1] // 480) 
            image = cv2.circle(image, (x1, y1), radius=image.shape[1] // 240, color=(0,0,255), thickness=-1)

        # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Output", 1280,720)
        # cv2.imshow("Output", image)
        # cv2.waitKey(1)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def main():
    parser = argparse.ArgumentParser(description='ClothDemo')
    parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to model')
    parser.add_argument('--center_model_path', type=str, default=None, help='Path to optional center model ')    
    args = parser.parse_args()

    demo = EcholibWrapper(ClothDemo(args.model_path, args.center_model_path))

    try:
        demo.run()
    except KeyboardInterrupt:
        pass

if __name__=='__main__':
    main()
