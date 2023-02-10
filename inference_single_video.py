# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2023-02-08
"""
import os
import torch
from sort import Sort
import random
import pathlib
from utils.general import increment_path, set_logging, check_img_size, check_imshow
from utils.torch_utils import select_device, TracedModel, load_classifier
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn




class InfVideo:
    def __init__(self):
        self.weights = 'yolov7.pt'
        self.download = True
        self.source = 'walking_people.mp4'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.nosave = False
        self.classes = [0]
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'Y7MOT'
        self.exist_ok = False
        self.no_trace = False
        self.colored_trk = False
        self.save_bbox_dim = False
        self.save_with_object_id = False

    def launch(self):
        with torch.no_grad():
            self.detect()

    @staticmethod
    def get_sort_tracker():
        sort_max_age = 5
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        return Sort(max_age=sort_max_age,
                    min_hits=sort_min_hits,
                    iou_threshold=sort_iou_thresh)

    @staticmethod
    def get_rand_color_ls():
        rand_color_list = []
        for i in range(0, 5005):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            rand_color = (r, g, b)
            rand_color_list.append(rand_color)

        return rand_color_list

    def detect(self):
        source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id = self.source, self.weights, self.view_img, self.save_txt, self.img_size, not self.no_trace, self.colored_trk, self.save_bbox_dim, self.save_with_object_id

        # save_img = not self.nosave and not source.endswith('.txt')
        save_img = True

        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = False

        sort_tracker = InfVideo.get_sort_tracker()
        rand_color_list = InfVideo.get_rand_color_ls()

        # increment run
        save_dir = pathlib.Path(
            increment_path(
                pathlib.Path(self.project) / self.name,
                exist_ok=self.exist_ok
            )
        )
        # save_dir = 'runs/detect/Y7MOT'
        (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True,
                                                                                     exist_ok=True)
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        # if trace:
        #     model = TracedModel(model, device, self.img_size)
        #
        # if half:
        #     model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            print('LoadImages')
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1


        for path, img, im0s, vid_cap in dataset:
            print(path)



if __name__ == '__main__':
    inf_video = InfVideo()
    inf_video.launch()
