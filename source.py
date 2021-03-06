import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
from kerasyolo3.yolo import YOLO
from neuralstyle.stylize import stylize
import tensorflow as tf

nst_flag = True
nst_frame = None

def split(img, n):
    sectors = []
    width = img.shape[1]//n
    if n >1:
        for i in range(n-1):
            sectors.append(img[:,i*width:(i+1)*width].copy())
        sectors.append(img[:,(i+1)*width:].copy())
    else:
        sectors.append(img)
    return sectors
def join(sectors):
    return np.concatenate(sectors, axis = 1)
def apply_single_transform(sector,transform, objects):
    if transform == 'const':
        return sector
    if transform == 'canny':
        gray = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    if transform == 'b&w':
        gray = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if transform == 'yolo':
        model = objects[transform]
        img2 = Image.fromarray(sector)
        img2 = model.detect_image(img2)
        return np.asarray(img2)
    if transform == 'nst':
        global nst_flag
        global nst_frame

        left_upper = (0.1,0.1)
        right_bottom = (0.5,0.5)
        if nst_flag:
            # default arguments
            CONTENT_WEIGHT = 5e0
            CONTENT_WEIGHT_BLEND = 1
            STYLE_WEIGHT = 5e2
            TV_WEIGHT = 1e2
            STYLE_LAYER_WEIGHT_EXP = 1
            LEARNING_RATE = 1e1
            BETA1 = 0.9
            BETA2 = 0.999
            EPSILON = 1e-08
            STYLE_SCALE = 1.0
            ITERATIONS = 2000
            VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
            POOLING = 'max'
            resize_fac = 1.0
            prev_shape = sector.shape[:2]
            content_image = cv2.resize(sector,(int(sector.shape[1]*resize_fac),int(sector.shape[0]*resize_fac)))
            style_images = [cv2.imread('videos/1-style.jpg'), ]
            initial = content_image
            initial_noiseblend = 0
            preserve_colors = None
            iterations = ITERATIONS
            content_weight = CONTENT_WEIGHT
            content_weight_blend = CONTENT_WEIGHT_BLEND
            style_weight = STYLE_WEIGHT
            style_layer_weight_exp = STYLE_LAYER_WEIGHT_EXP
            style_blend_weights = [1,]
            tv_weight = TV_WEIGHT
            learning_rate = LEARNING_RATE
            beta1 = BETA1
            beta2 = BETA2
            epsilon = EPSILON
            pooling = POOLING
            print_iterations= None
            checkpoint_iterations = None
            for iteration, image, loss_vals in stylize(
                                                network=VGG_PATH,
                                                initial=initial,
                                                initial_noiseblend=initial_noiseblend,
                                                content=content_image,
                                                styles=style_images,
                                                preserve_colors=preserve_colors,
                                                iterations=iterations,
                                                content_weight=content_weight,
                                                content_weight_blend=content_weight_blend,
                                                style_weight=style_weight,
                                                style_layer_weight_exp=style_layer_weight_exp,
                                                style_blend_weights=style_blend_weights,
                                                tv_weight=tv_weight,
                                                learning_rate=learning_rate,
                                                beta1=beta1,
                                                beta2=beta2,
                                                epsilon=epsilon,
                                                pooling=pooling,
                                                print_iterations=print_iterations,
                                                checkpoint_iterations=checkpoint_iterations,
            ):
                pass
            min_v = np.min(image)
            image = image-min_v #to have only positive values
            max_v=np.max(image) 
            div=max_v/255.0 #calculate the normalize divisor
            image=np.uint8(image/div)
            nst_frame = cv2.resize(image, (prev_shape[1], prev_shape[0]))
            nst_flag = False
            return cv2.resize(image, (prev_shape[1], prev_shape[0]))
        else:
            dur = objects['dur']
            if dur>=1.0:
                dur = 1.0
            if dur<=0.0:
                dur = 0.0
            lu_rel = np.array(left_upper)*dur
            rb_rel = 1 - np.array(right_bottom)*dur
            pixels_lu = (lu_rel*np.array([sector.shape[1], sector.shape[0]])).astype(np.uint32)
            pixels_rb = (rb_rel*np.array([sector.shape[1], sector.shape[0]])).astype(np.uint32)
            print(dur)
            print(lu_rel, rb_rel)
            print(pixels_lu, pixels_rb)
            sector[pixels_lu[1]:pixels_rb[1],pixels_lu[0]:pixels_rb[0], :] = cv2.resize(nst_frame, (pixels_rb[0]-pixels_lu[0], pixels_rb[1]-pixels_lu[1]))
            return sector
        
def apply_transforms(frame, keep_orig, transforms, objects):
    if keep_orig:
        sectors = split(frame, len(transforms)+1)
        transforms = ['const']+transforms
    else:
        sectors = split(frame, len(transforms))
    sectors_transformed = [apply_single_transform(sector, transform, objects) for sector,transform in zip(sectors,transforms)]
    frame_transformed = join(sectors_transformed)
    return frame_transformed
def pre_prepare_video(video_path):
    os.system('ffmpeg -i '+'videos/afonso.mp4' + ' -vn ' + 'out/test.mp3')
def check_weights():
    if not os.path.isfile('kerasyolo3/model_data/yolo.h5'):
        print("Weights not downloaded. Downloading now!")

        os.system('cd kerasyolo3/ && wget https://pjreddie.com/media/files/yolov3.weights && python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5')
    else:
        print("Weights alerady downloaded.")
if __name__ == "__main__":
    check_weights()
    model = YOLO()
    parameters = [
        {'duration':(1800, 1950), 'keep_orig':True, 'transforms':['b&w', 'canny'], 'transform_objects':{}},
        {'duration':(1985, 2200), 'keep_orig':False, 'transforms':['yolo'], 'transform_objects':{"yolo": model}},
        {'duration':(2264, 2400), 'keep_orig':False, 'transforms':['nst'], 'transform_objects':{"dur": -2}},
        {'duration':(2527, 2675), 'keep_orig':False, 'transforms':['nst'], 'transform_objects':{"dur": -2}},
        #{'duration':(700, 1200), 'keep_orig':False, 'transforms':['canny', 'const'], 'transform_objects':{}},
    ]
    
    
    for parameter in parameters:
        if 'yolo' in parameter['transforms']:
            foo = {"yolo":model}
            parameter['transform_objects'].update(foo)

    pre_prepare_video('videos/afonso.mp4')
    cap = cv2.VideoCapture('videos/afonso.mp4')
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('width = {} height = {} fps = {}'.format(width, height, fps))
    out = cv2.VideoWriter('out/output.avi',fourcc, 23.98, (1280,720))
    while(True):
        ret, frame = cap.read()
        if ret:
            
            frame_count += 1
            #if(frame_count%100 == 0):
            print(frame_count)
            # print(frame.shape)
            # sectors = split(frame, 3)
            # sectors = [apply_single_transform(sector, transform) for sector,transform in zip(sectors,['const', 'b&w', 'canny'])]
            # for i, sector in enumerate(sectors):
            #     print(i)
            #     cv2.imshow('sector {}'.format(i), sector)
            # cv2.imshow('join', join(sectors))
            cv2.imwrite("videos/framoso.png", frame)
            for parameter in parameters:
                if frame_count>=parameter['duration'][0] and frame_count<=parameter['duration'][1] and parameter['transforms'] == ['nst']:
                    parameter['transform_objects']["dur"] += 0.1
                if frame_count>=parameter['duration'][0] and frame_count<=parameter['duration'][1]:
                    frame = apply_transforms(frame, keep_orig = parameter['keep_orig'], transforms = parameter['transforms'], objects = parameter['transform_objects'])   
                if frame_count == parameter['duration'][1] and parameter['transforms'] == ['nst']:
                    nst_flag = True
                
            out.write(frame)
        
            # cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.system('ffmpeg -i '+'out/output.avi'+' -i '+'out/test.mp3' +' -codec copy -shortest '+'out/output_a.avi')