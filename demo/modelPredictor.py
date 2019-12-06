# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import time
from lxml import etree, objectify
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from bs4 import BeautifulSoup

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

class Predictor(object):

    def __init__(self, label_map):
        self.label_map = label_map
        pass

    def initModel(
        self,
        cfg_path,
        wts_path,
    ):
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        print('load wts from:', wts_path)
        _ = checkpointer.load(wts_path, use_latest=False)

        self.transforms = self.build_transform()
        self.cpu_device = torch.device("cpu")

    def getInfoByModel(self,image,thresh=0.7):
        #np.set_printoptions(suppress=True) #numpy不以科学计数法输出
        col = float(image.shape[1])
        row = float(image.shape[0])
        res = None
        
        predictions = self.compute_prediction(image)      
        top_predictions = self.select_top_predictions(predictions,thresh)
        get_bbox=top_predictions.bbox.numpy()
        if get_bbox is not None:
            get_bbox=get_bbox/np.array([col,row,col,row])
            get_label=top_predictions.get_field("labels").numpy()
            get_scores=top_predictions.get_field("scores").numpy()
            get_label = np.array(get_label,dtype=int)
            get_label = np.transpose(get_label.reshape(1,-1))
            res = np.hstack((get_label,get_bbox))
            get_scores = np.transpose(get_scores.reshape(1,-1))
            res = np.hstack((res,get_scores))
            # print(get_bbox)
            # print('==================')
            # print(get_label)
            # print('==================')
            # print(get_scores)
            # print('==================')
            # print(res)
        else:
            res=[]
        return res

    def build_transform(self):
        cfg = self.cfg

        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_prediction(self, original_image):
        #1
        image = self.transforms(original_image)       
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image.to(self.device) 
        #1 12ms

        # compute predictions
        #2
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        #2 50ms

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        return prediction

    def select_top_predictions(self, predictions,thresh):
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def create_new_anno(self,file_name, width,height):
        w=width
        h=height
        name=str(file_name)
        
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('a'),
            E.filename(name),
            E.source(
                E.database('oil'),
            ),
            E.size(
                E.width(w),
                E.height(h),
                E.depth(3)
            ),
            E.segmented(0)
        )
        
        return anno_tree
        
        
    def create_new_obj(self,obj_name,x1,y1,x2,y2):
        rec_name= str(obj_name)
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E2.name(rec_name),
            E2.bndbox(
                E2.xmin(x1),
                E2.ymin(y1),
                E2.xmax(x2),
                E2.ymax(y2)
            ),
            E2.difficult(0)
        )
        return anno_tree2

    def get_xml_result(self, image, name):
        label_map= self.label_map
        h=image.shape[0]
        w=image.shape[1]
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions, 0.7)
        get_bbox=top_predictions.bbox.numpy()
        get_label=top_predictions.get_field("labels").numpy()
        obj_num=len(get_bbox)
        tree=self.create_new_anno(name,w,h)
        for i in range(obj_num):
            obj_name=label_map[get_label[i]]
            xmin=int(get_bbox[i][0])
            ymin=int(get_bbox[i][1])
            xmax=int(get_bbox[i][2])
            ymax=int(get_bbox[i][3])
            obj_tree=self.create_new_obj(obj_name,xmin,ymin,xmax,ymax)
            tree.append(obj_tree)
        return tree,top_predictions
        
    def get_image_result(self, image, gt, top_predictions):
        result = image.copy()
        if gt:
            # gt
            result = self.overlay_boxes(result, gt, True)
            result = self.overlay_class_names(result, gt, True)
        # pre
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        return result


    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        
        colors = labels[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions, is_gt=False):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        
        if is_gt:
            palette = torch.tensor([0, 0, 255])
            labels = torch.tensor(np.ones_like(labels))
            colors = labels[:, None] * palette
            colors = colors.numpy().tolist()
            #print(colors)
        else:
            colors = self.compute_colors_for_labels(labels).tolist()
            #print(colors)

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image
        
        
    def overlay_class_names(self, image, predictions, is_gt=False):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.label_map[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            if is_gt:
                s = "GT:{}".format(label)
                cv2.putText(
                    image, s, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            else:
                s = template.format(label, score)
                cv2.putText(
                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        return image 

    def get_gt(self, path):

        objs, size = self.read_objects(path)
        #print(objs)
        box_list = []
        label_list = []
        score_list = []
        for obj in objs:
            name = obj['name']
            idx = [k for k,v in self.label_map.items() if v == name][0]
            #print(name, idx)
            xmin = int(obj['xmin'])
            ymin = int(obj['ymin'])
            xmax = int(obj['xmax'])
            ymax = int(obj['ymax'])
            box_list.append([xmin,ymin,xmax,ymax])
            label_list.append(idx)
            score_list.append(1)
        
        #print(label_list)
        if box_list==[]:
            #print(path)
            return None
        
        gt_bbox = BoxList(box_list, size)
        gt_bbox.add_field('labels', torch.as_tensor(label_list))
        gt_bbox.add_field('scores', torch.as_tensor(score_list))
        return gt_bbox

    def read_objects(self, path=None):
        if path==None:
            print("[ERROR ] Path is None!")
            return []
        objs=[]
        with open(path,'rb') as fr:
            all_txt=fr.read()
            soup=BeautifulSoup(all_txt,"lxml")
            s_objects=soup.find('size')
            siz_width=s_objects.find("width").text
            siz_height=s_objects.find("height").text
            #print(siz_width, siz_height)
            m_objects=soup.find_all('object')
            for m_object in m_objects:
                name=m_object.find_all("name")[0].get_text()
                xmin=m_object.find_all("xmin")[0].get_text()
                ymin=m_object.find_all("ymin")[0].get_text()
                xmax=m_object.find_all("xmax")[0].get_text()
                ymax=m_object.find_all("ymax")[0].get_text()
                objs.append({"name":name,
                    "xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax})
        return objs, (siz_width, siz_height)

