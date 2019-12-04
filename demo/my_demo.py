# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
from maskrcnn_benchmark.config import cfg
from my_predictor import COCODemo

import time

vis = True

def main():

    config_file = '/media/kevin/办公/xizang/1119_outputs/1119_infer_configs.yaml'
    opts = ["MODEL.DEVICE", "cuda"]
    confidence_threshold = 0.7
    min_image_size = 720

    # load config from file and command-line arguments
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    print(cfg)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=confidence_threshold,
        min_image_size=min_image_size,
    )

    img_dir = '/media/kevin/娱乐/xizang_database/testdata/1118/JPEGImages'
    img_out_dir = '/media/kevin/办公/xizang/主观结果/1119'
    img_list = os.listdir(img_dir)              # 获取类别文件夹下所有图片的路径
    
    for i in range(0, 1500, 50):
    
        if not img_list[i].endswith('jpg'):         # 若不是jpg文件，跳过
            continue
        # print(i)
        start_time = time.time()
        img_path = os.path.join(img_dir, img_list[i])
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 720))
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        img_out_path = os.path.join(img_out_dir, 'det_'+img_list[i])
        cv2.imwrite(img_out_path, composite)
        print('image write to %s'%img_out_path)
        
        
        if vis:
            cv2.namedWindow("COCO detections",0);
            cv2.resizeWindow("COCO detections", 1800, 1100);
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(5000) == 27:
                exit()
        
    if vis:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
