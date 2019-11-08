from model import *
import os
import time
from BackgroundParser import SourcePreparation
from keras.models import load_model
#from tensorflow.keras.models import load_model
import cv2


current_dir = os.getcwd()
model_path = current_dir+'/models/unet2019-11-04-02-14-04.13-tloss-0.1455-tdice-0.8545-vdice-0.6879.hdf5'
result_path = current_dir+"/dataset/results_2019-01-31-15-53-26_kia_velo_gps_time_no_argmax"
#test_path = "D:/Projects/nkbvs_segmentation/dataset/augmented_multiclass_dataset/test/color"
test_path = "D:/Datasets/NKBVS/mfti_data/2019-01-31-15-53-26_kia_velo_gps_time/stereo/left/image_raw/converted"
is_armax = True
is_save_crop = True

# detect_dict = {"traffic_sign": 0,
#               "traffic_light": 6}
detect_dict = {
               "traffic_light": 6} #label name: index of result channel
class_dict = {"0": 'off',
              "1": "r",
              "2": "y",
              "3": "ry",
              "4": "g",
              "5": "no"}

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

model_t_l_classify = load_model('models/Best_ResNetM_50_epochs_TrafficLights_without_Dense.hdf5')



def read_and_preprocess_image(test_path,filename,target_size = (256,256)):

    raw = cv2.imread(os.path.join(test_path, filename))
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = raw / 255
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.reshape(img, (1,) + img.shape)
    scale_coef = (raw.shape[0]/target_size[0], raw.shape[1]/target_size[1])

    return img, raw, scale_coef

def preprocess_image(raw,target_size = (256,256)):
    #raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = raw / 255
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.reshape(img, (1,) + img.shape)

    return img

def convert_str_to_rgb(str_value):
    return (int(str_value[4:6], 16), int(str_value[2:4], 16), int(str_value[0:2], 16)) #in bgr format

os.makedirs(result_path, exist_ok=True)

for filename in sorted(os.listdir(test_path)):

    img, raw, scale_coef = read_and_preprocess_image(test_path = test_path,
                                    filename = filename,
                                    target_size = (image_height, image_width))
    start_time = time.time()
    results = model.predict(img,verbose=1)
    end_time = time.time()
    duration = end_time - start_time
    # with open(log_filename, "a") as log_file:
    #     log_file.write("Testing time, sec: " + str(duration) + "\n")

    out_file_name = os.path.join(result_path, filename)


    src_img = img[0]*255

    if is_save_crop:
        src_img_to_crop = raw
        os.makedirs(os.path.join(result_path, 'crop'),exist_ok=True)

    if is_armax:
        results_argmax = np.argmax(results[0, :, :], axis=2)
    bboxes = []
    for label in detect_dict:
        idx = detect_dict[label]
        label_color = convert_str_to_rgb(color_mask_dict[label_list[idx]])
        if is_armax:
            mask = np.zeros((results.shape[1], results.shape[2]), dtype=np.uint8)
            mask[results_argmax==idx] = 255
        else:
            mask = results[0,:, :, idx]
            mask[mask > 0.5] = 255
            mask[mask <= 0.5] = 0

        src_img[mask == 255] *= np.array(label_color)/255
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
                if is_save_crop:
                    crop_to_save = src_img_to_crop[
                                   int(top_left_corner[0]*scale_coef[0]):int(bottom_right_corner[0]*scale_coef[0]),
                                   int(top_left_corner[1]*scale_coef[1]):int(bottom_right_corner[1]*scale_coef[1]),
                                   :]
                    input_img = preprocess_image(crop_to_save.astype(dtype=np.uint8), target_size=(70, 70))
                    class_result = np.argmax(model_t_l_classify.predict(input_img, verbose=1))
                    class_text = class_dict[str(class_result)]

                    cv2.putText(src_img, class_text, (top_left_corner[1], top_left_corner[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 1, cv2.LINE_AA)

                    os.makedirs(result_path + "/crop/"+class_text, exist_ok=True)
                    crop_to_save = cv2.cvtColor(crop_to_save.astype(dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(result_path + "/crop/"+class_text+"/"+filename+'_'+str(label)+'_'+str(region_id)+'.png', crop_to_save)
    src_img = cv2.cvtColor(src_img.astype(dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file_name + "_detect.png", src_img)







