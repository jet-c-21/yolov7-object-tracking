# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2023-02-08
"""
import os
import numpy as np
import cv2
import torch
from sort import Sort
import random
import pathlib
from utils.general import increment_path, set_logging, check_img_size, check_imshow, non_max_suppression, \
    apply_classifier, scale_coords
from utils.torch_utils import select_device, TracedModel, load_classifier, time_synchronized
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

    @staticmethod
    def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,
                   offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
            label = str(id) + ":" + names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, [255, 255, 255], 1)
            # cv2.circle(img, data, 6, color,-1)   #centroid of box
            txt_str = ""
            if save_with_object_id:
                txt_str += "%i %i %f %f %f %f %f %f" % (
                    id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                    int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                    int(box[1] + (
                            box[3] * 0.5)) / img.shape[0])
                txt_str += "\n"
                with open(path + '.txt', 'a') as f:
                    f.write(txt_str)
        return img

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

        if trace:
            model = TracedModel(model, device, self.img_size)

        if half:
            model.half()  # to FP16

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
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = pathlib.Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # ..................USE TRACK FUNCTION....................
                    # pass an empty array to sort
                    dets_to_sort = np.empty((0, 6))

                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort,
                                                  np.array([x1, y1, x2, y2, conf, detclass])))

                    # Run SORT
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()

                    txt_str = ""

                    # loop over tracks
                    for track in tracks:
                        # color = compute_color_for_labels(id)
                        # draw colored tracks
                        if colored_trk:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                      (int(track.centroidarr[i + 1][0]),
                                       int(track.centroidarr[i + 1][1])),
                                      rand_color_list[track.id], thickness=2)
                             for i, _ in enumerate(track.centroidarr)
                             if i < len(track.centroidarr) - 1]
                            # draw same color tracks
                        else:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                      (int(track.centroidarr[i + 1][0]),
                                       int(track.centroidarr[i + 1][1])),
                                      (255, 0, 0), thickness=2)
                             for i, _ in enumerate(track.centroidarr)
                             if i < len(track.centroidarr) - 1]

                        if save_txt and not save_with_object_id:
                            # Normalize coordinates
                            txt_str += "%i %i %f %f" % (
                                track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1],
                                track.centroidarr[-1][1] / im0.shape[0])
                            if save_bbox_dim:
                                txt_str += " %f %f" % (
                                    np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0],
                                    np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                            txt_str += "\n"

                    if save_txt and not save_with_object_id:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(txt_str)

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        InfVideo.draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
                    # ........................................................

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cv2.destroyAllWindows()
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # print(f"video_save_path = {save_path}")
                        vid_writer.write(im0)


if __name__ == '__main__':
    inf_video = InfVideo()
    inf_video.launch()
