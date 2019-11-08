from model import *
#from data import *
#from keras import callbacks
#from keras.callbacks import ModelCheckpoint
#from datetime import datetime
import os
import time
from BackgroundParser import SourcePreparation
import cv2


current_dir = os.getcwd()
model_path = current_dir+'/models/unet2019-11-04-02-14-04.13-tloss-0.1455-tdice-0.8545-vdice-0.6879.hdf5'
result_path = current_dir+"/dataset/results_2019-01-31-15-53-26_kia_velo_gps_time"
#test_path = "D:/Projects/nkbvs_segmentation/dataset/augmented_multiclass_dataset/test/color"
test_path = "D:/Datasets/NKBVS/mfti_data/2019-01-31-15-53-26_kia_velo_gps_time/stereo/left/image_raw/converted"
image_width = 640
image_height = 384
batch_size = 1
# timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
# log_filename = current_dir+'/logs/log-unet-'+ timestr +'.txt'

source_preparator = SourcePreparation()
label_list = source_preparator.label_list
color_mask_dict = source_preparator.color_mask_dict
num_class = len(label_list)

model = unet_light_ct_tv(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
                       n_classes=num_class)
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

# with open(log_filename, "a") as log_file:
#     log_file.write('\nModel summary:\n')
#     log_file.write(short_model_summary)
#     log_file.write('Loss:' + str(model.loss) + '\n')
#     log_file.write('Metrics:' + str(model.metrics) + '\n')
#     log_file.write('Image_Size: ' + str(image_width) + ' x '+str(image_height) + '\n')
#     log_file.write('Batch_Size: ' + str(batch_size) + '\n')
#     log_file.write('\nTraining process:\n')
#     log_file.write('\nEpoch no, train dice, train loss, val dice, val loss\n')

def read_and_preprocess_image(test_path,filename,target_size = (256,256)):
    start_time = time.time()
    img = cv2.imread(os.path.join(test_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.reshape(img, (1,) + img.shape)
    end_time = time.time()
    duration = end_time - start_time
    # with open('logs/timeread.txt', "a") as log_file:
    #     log_file.write("Imread, s: " + str(duration) + "\n")
    return img

def labelVisualize(label_list, color_mask_dict, img, save_path, id_img):
    num_class = len(label_list)
    img_out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(num_class):
        idx = num_class-1-i
        label_color = convert_str_to_rgb(color_mask_dict[label_list[idx]])
        mask = img[:,:,idx]#/np.max(img[:,:,idx])
        img_out[mask > 0.5] = np.array(label_color)

    return img_out

def convert_str_to_rgb(str_value):
    return (int(str_value[4:6], 16), int(str_value[2:4], 16), int(str_value[0:2], 16)) #in bgr format

def saveResult(save_path,npyfile,flag_multi_class = False,label_list=None,color_mask_dict=None):
    for i,item in enumerate(npyfile):
        img = labelVisualize(label_list,color_mask_dict, item,save_path, i) if flag_multi_class else item[:,:,0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path+"_predict%03d.png"%i,img)

os.makedirs(result_path, exist_ok=True)
for filename in sorted(os.listdir(test_path)):
    img = read_and_preprocess_image(test_path = test_path,
                                    filename = filename,
                                    target_size = (image_height, image_width))
    start_time = time.time()
    results = model.predict(img,verbose=1)
    end_time = time.time()
    duration = end_time - start_time
    with open(log_filename, "a") as log_file:
        log_file.write("Testing time, sec: " + str(duration) + "\n")

    out_file_name = os.path.join(result_path, filename)
    # saveResult(out_file_name, results, flag_multi_class=True,
    #            label_list=label_list, color_mask_dict=color_mask_dict)

    #detect_list = ["traffic_sign",
    #               "traffic_light"]
    detect_dict = {"traffic_sign": 0,
                  "traffic_light": 6}
    src_img = img[0]*255
    img_out = np.zeros((results.shape[1], results.shape[2], 3), dtype=np.uint8)
    bboxes = []
    for label in detect_dict:
        idx = detect_dict[label]
        label_color = convert_str_to_rgb(color_mask_dict[label_list[idx]])
        mask = results[0,:, :, idx]
        #img_out[mask > 0.5] = 1
        mask[mask > 0.5] = 255
        mask[mask < 0.5] = 0
        src_img[mask > 0.5] *= np.array(label_color)/255
        ret, labels = cv2.connectedComponents(mask.astype(dtype=np.uint8))
        area_threshold = 30
        for region_id in range(ret-1):
            # region_id==0 is background
            region_points = np.argwhere(labels==(region_id+1))#labels[labels==region_id]
            region_area = len(region_points)
            if(region_area > area_threshold):
                top_left_corner = region_points.min(axis = 0)
                bottom_right_corner = region_points.max(axis = 0)
                bboxes.append({"label": label,
                               "top_left_corner": top_left_corner,
                               "bottom_right_corner": bottom_right_corner})
                cv2.rectangle(src_img, (top_left_corner[1], top_left_corner[0]),
                              (bottom_right_corner[1], bottom_right_corner[0]), label_color, 1)
    src_img = cv2.cvtColor(src_img.astype(dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file_name + "_detect.png", src_img)







