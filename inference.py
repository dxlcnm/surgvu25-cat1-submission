"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
import cv2
import os
from ultralytics import YOLO
from timm.models import create_model
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import logging
from ultralytics.utils import LOGGER
from torchvision.ops import nms

import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--input_path", type=str, default="/input", help="input path")
    parser.add_argument("--output_path", type=str, default="/output", help="output path")
    args = parser.parse_args()
    return args


def load_json_file(*, location):
        # Reads a json file
        with open(location, "r") as f:
            return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('inputs - ', inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    print('socket slugs', socket_slugs)
    return tuple(sorted(socket_slugs))


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class Inference(object):
    def __init__(self, det_model_path, cls_model_path="", debug=False):
        self.id_label_dict = {0: 'needle_driver', 
                        1: 'monopolar_curved_scissors', 
                        2: 'force_bipolar', 
                        3: 'clip_applier', 
                        4: 'cadiere_forceps', 
                        5: 'bipolar_forceps', 
                        6: 'vessel_sealer', 
                        7: 'permanent_cautery_hook_spatula',
                        8: 'prograsp_forceps', 
                        9: 'stapler', 
                        10: 'grasping_retractor', 
                        11: 'tip-up_fenestrated_grasper'}
        num_classes = len(self.id_label_dict.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.det_model = YOLO(det_model_path)
        self.cls_model = None
        if cls_model_path!="":
            self.cls_model = create_model(
            "resnet50",
            pretrained=False,
            num_classes=num_classes,
            )

            checkpoint = torch.load(cls_model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint['model']
            load_state_dict(self.cls_model, state_dict)
            self.cls_model.to(self.device)
            self.cls_model.eval()
        self.cls_transform = transforms.Compose([transforms.Resize(size=(224, 224)), 
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                ])
        self.debug = debug

    
    def run(self):
        # The key is a tuple of the slugs of the input sockets
        interface_key = get_interface_key()
        LOGGER.setLevel(logging.ERROR)  # 只显示错误

        # Lookup the handler for this particular set of sockets (i.e. the interface)
        handler = {
            ("endoscopic-robotic-surgery-video",): self.interf0_handler,
        }[interface_key]

        # Call the handler
        return handler()
    
    def nms_per_class(self, classified_boxes, iou_threshold, per_class=True):
        """
        nus
        Parameters
            classified_boxes: instrument detection res with cls, like the format of [[xmin, ymin, xmax, ymax, prob, cls], ...]
            iou_threshold: iou threshold
            per_class: do nms each class or not
        Returns: 
            classified_boxes: remained boxes adter nms
        """
        if len(classified_boxes) == 0:
            return classified_boxes
        boxes = []
        scores = []
        labels = []
        for i, box in enumerate(classified_boxes):
            boxes.append([box[0], box[1], box[2], box[3]])
            scores.append(box[4])
            labels.append(box[5])
        boxes = torch.Tensor(boxes)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        if per_class:
            unique_labels = labels.unique()
            keep_indices = []
            for label in unique_labels:
                index = torch.nonzero(labels == label, as_tuple=True)[0][nms(boxes[labels == label], scores[labels == label], iou_threshold)]
                if len(index) > 0:
                    keep_indices.append(index)
            if len(keep_indices) > 0:
                keep_indices = torch.cat(keep_indices)
        else:
            keep_indices = nms(boxes, scores, iou_threshold)
        remained_boxes = [classified_boxes[i] for i in keep_indices]
        return remained_boxes


    def interf0_handler(self):
        video_path = os.path.join(INPUT_PATH, "endoscopic-robotic-surgery-video.mp4")
        # Read the input
        cap = cv2.VideoCapture(video_path)
        frame_idx = -1
        
        output_boxes = []
        while cap.isOpened():
            success, frame = cap.read()
            
            # Break the loop if the end of the video is reached
            if not success:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            det_boxes = []
            # Run YOLO11 on the frame 
            # det_res = self.det_model(frame)[0]
            det_res = self.det_model.predict(frame, conf=0.25, iou=0.7)[0]
            # ori_img = det_res.orig_img
            boxes = det_res.boxes.xyxy.cpu().numpy().astype(np.int32)
            clses = det_res.boxes.cls.cpu().numpy().astype(np.int32)
            confs = det_res.boxes.conf.cpu().numpy()
            id_names = det_res.names
            for i in range(len(boxes)):
                det_boxes.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], confs[i], clses[i]])
                
            # classification
            if len(det_boxes) == 0:
                continue
            for i, box in enumerate(det_boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bw, bh = x2-x1, y2-y1
                det_conf = float(box[4])
                det_cls = box[5]
                det_label = self.id_label_dict[det_cls]
                det_label_dbg = det_label

                cls_label = " "
                if self.cls_model != None:
                    # resnet cla result
                    h, w = frame.shape[0], frame.shape[1]
                    h_start = 0
                    w_start = int(190 / 1280 * w)

                    # expansion twice
                    bw, bh = x2-x1, y2-y1
                    xmin, ymin, xmax, ymax = x1,y1,x2,y2
                    xmin, ymin = max(xmin - bw //2, w_start), max(ymin-bh//2, h_start)
                    xmax, ymax = min(xmax+bw//2, w), min(ymax+bh//2, h)
                    if xmin < xmax and ymin < ymax:
                        crop = frame[ymin: ymax, xmin: xmax, :]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_rgb = Image.fromarray(crop_rgb)
                    
                        image_trans = self.cls_transform(crop_rgb)
                        image_trans = image_trans.unsqueeze(0)
                        image_trans = image_trans.to(self.device)
                        res = self.cls_model(image_trans)
                        output = torch.softmax(res, dim=1)
                        output = output.cpu().detach().numpy()
                        cls_id = np.argmax(output, axis=1)[0]
                        cls_label = self.id_label_dict[cls_id]
                        cls_conf = output[0][cls_id]
                        # replace det label with cls label
                        det_label = cls_label
                        det_boxes[i][5] = cls_id
                    
            # do nms after clssification [x1, y1, x2, y2, conf, cls]
            # det_boxes = self.nms_per_class(det_boxes, iou_threshold=0.001, per_class=True)
            for det_box in det_boxes:
                x1,y1,x2,y2,det_conf,det_label = int(det_box[0]),int(det_box[1]),int(det_box[2]),int(det_box[3]),float(det_box[4]),int(det_box[5])
                bw, bh = x2 - x1, y2 - y1
                one_box = {"name": f"slice_nr_{frame_idx}_{self.id_label_dict[det_label]}",
                           "corners": [[x1, y1, 0.5],
                                       [x1+bw, y1, 0.5],
                                       [x1+bw, y1+bh, 0.5],
                                       [x1, y1+bh, 0.5]],
                           "probability": det_conf}
                output_boxes.append(one_box)
                    
            if self.debug:
                for box in det_boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    label = f"cls:{box[5]},p: {str(round(float(box[4]),2))}"
                    cv2.putText(frame, label, (int(box[0]), int(box[3]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                os.makedirs(os.path.join(OUTPUT_PATH, "debug"), exist_ok=True)
                cv2.imwrite(os.path.join(OUTPUT_PATH, "debug", f"frame_{str(frame_idx).zfill(5)}.jpg"), frame)       
            
        # For now, let us make bogus predictions
        output_surgical_tools = {
            "name": "Regions of interest",
            "type": "Multiple 2D bounding boxes",
            "boxes": output_boxes,
            "version": {
                "major": 1,
                "minor": 0
            }
        }
        
        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "surgical-tools.json", content=output_surgical_tools
        )
        print('json file generated by the submission container')

        return 0

if __name__ == "__main__":
    # example: python inference.py --input_path test/input/interf0 --output_path test/output/interf0
    args = parse_arguments()
    INPUT_PATH = Path(args.input_path)
    OUTPUT_PATH = Path(args.output_path)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    det_model_path = "model/surgvu25-cat1/last.pt"
    cls_model_path = "model/surgvu25-cat1/checkpoint-399.pth"
    infer = Inference(det_model_path, cls_model_path, False)
    raise SystemExit(infer.run())
