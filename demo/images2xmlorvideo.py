from modelPredictor import Predictor
import cv2
import glob
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import argparse
from lxml import etree, objectify

parser = argparse.ArgumentParser(description='images to xml by model')
parser.add_argument('--imgs-dir',type=str,default='./imgs')
parser.add_argument('--outputs-dir',type=str,default='./outputs')
parser.add_argument('--scene',type=str,default='jt')
parser.add_argument('--wscene',type=str,default=None)
parser.add_argument('--cfg-file',type=str,default=None)
parser.add_argument('--wts-file',type=str,default=None)
parser.add_argument('--label',type=str,default='jt2xizang')
args=parser.parse_args()

cfg_dic={
	'jt':'/media/kevin/办公/xizang/JT_model/models/model_one/e2e_faster_rcnn_R_50_FPN_1x.yaml',
	'xizang':'/media/kevin/办公/xizang/1130_ex18_outputs/infer_configs.yaml',
	'nofe':'/media/kevin/办公/NofE/1203_ex6_outputs/infer_configs.yaml',
}

wts_dic={
	'jt':'/media/kevin/办公/xizang/JT_model/models/model_one/model_final.pth',
	'xizang':'/media/kevin/办公/xizang/1130_ex18_outputs/model_final.pth',
	'nofe':'/media/kevin/办公/NofE/1203_ex6_outputs/model_final.pth',
}

label_dic={
    'jt':{
    1:'person',
    2:'motorbike',
    3:'car',
    4:'bus',
    5:'truck',
    6:'microbus',
    7:'pickup',
    8:'SUV',
    9:'tanker',
    10:'tractor',
    11:'engineeringvan',
    12:'tricycle',
},
    'xizang4classes':{
    1: "person_foreign",
    2: "car_foreign",
    3: "person",
    4: "car"
},
    'xizang12classes':{
    1: "person_foreign",
    2: "motorbike_foreign",
    3: "car_foreign",
    4: "bus_foreign",
    5: "truck_foreign",
    6: "microbus_foreign",
    7: "pickup_foreign",
    8: "SUV_foreign",
    9: "tanker_foreign",
    10: "tractor_foreign",
    11: "engineeringvan_foreign",
    12: "tricycle_foreign"
},
    'jt2xizang':{
    1: "person_foreign",
    2: "car_foreign",
    3: "car_foreign",
    4: "car_foreign",
    5: "car_foreign",
    6: "car_foreign",
    7: "car_foreign",
    8: "car_foreign",
    9: "car_foreign",
    10: "car_foreign",
    11: "car_foreign",
    12: "car_foreign"
},
    'nofe': {
    1: "car",
    2: "bus",
    3: "van",
    4: "others",
    },
}

scene = args.scene
print('当前选择模型场景为：{} \n'.format(scene))
if args.cfg_file:
    cfg_path=args.cfg_file
else:
    cfg_path=cfg_dic[scene]
print('当前选择配置文件路径为：{} \n'.format(cfg_path))
if args.wts_file:
    wts_path=args.wts_file
else:
    wts_path=wts_dic[scene]
print('当前选择权重文件路径为：{} \n'.format(wts_path))
if args.label:
    label_map=label_dic[args.label]
else:
    label_map=label_dic[scene]
print('当前选择映射标签表为：{} \n'.format(label_map))
img_path=args.imgs_dir
out_xml_path=os.path.join(args.outputs_dir, 'xml')
if not os.path.exists(out_xml_path):
    os.makedirs(out_xml_path)
img_out_dir = os.path.join(args.outputs_dir, "img")
if not os.path.exists(img_out_dir):
    os.mkdir(img_out_dir)
print('原始图片路径为：{} \nxml输出路径为：{}\nimg输出路径为：{}'.format(img_path,out_xml_path,img_out_dir))

detectModel=Predictor(label_map)
detectModel.initModel(cfg_path, wts_path)
print('model init finish!')

need_xml = False
need_img = True
need_video = False


if need_video:
    # init video info
    video_name='outputs'
    # fps
    video_fps = 24
    # resize
    video_size = (1920, 1080) 
    #size = (1280, 720)        
            
    # point out how to encode videos
    # I420-avi=>cv2.cv.CV_FOURCC('X','2','6','4');
    # MP4=>cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    result_path = os.path.join(args.outputs_dir, video_name + "_det.avi")
    videowriter = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), video_fps, video_size)

if args.wscene != None or args.wscene != 'all':
    im_list = glob.glob(img_path + '/{}*.jpg'.format(args.wscene))
else:
    im_list = glob.glob(img_path + '/*.jpg')
print(im_list[:10])
pic_len=len(im_list)


for i in tqdm(range(pic_len)):
    if i%50 != 0:
        continue
    pic_name=os.path.basename(im_list[i])
    image_in = cv2.imread(im_list[i])

    if need_video:
        # 对video需要resize,对xml不需要
        image_in = cv2.resize(image_in, (video_size[0], video_size[1]))
    
    '''
    cv2.imshow('in', image_in)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    '''
	
    get_xml,top_predictions = detectModel.get_xml_result(image_in,pic_name)

    if need_xml:
        etree.ElementTree(get_xml).write(os.path.join(out_xml_path,pic_name).rstrip(".jpg") + '.xml', pretty_print=True)

    if need_img or need_video:
        result = detectModel.get_image_result(image_in, top_predictions)  
        '''
        cv2.imshow('out', result)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        '''
        
    if need_img:
        out_path = os.path.join(img_out_dir, pic_name)
        cv2.imwrite(out_path, result)  # write one frame into the output img

    if need_video:
        videowriter.write(result)  # write one frame into the output video

if need_video:
    videowriter.release()
    
print('done!')

