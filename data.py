from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import cv2
import time

def convert_str_to_rgb(str_value):
    return (int(str_value[4:6], 16), int(str_value[2:4], 16), int(str_value[0:2], 16)) #in bgr format

def adjustData(img,mask,flag_multi_class,num_class,label_list=None,color_mask_dict=None):
    if(flag_multi_class):
        img = img / 255

        new_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], num_class))
        for i in range(num_class):
            label_color = convert_str_to_rgb(color_mask_dict[label_list[i]])

            min_color = (label_color[0] - 5, label_color[1] - 5, label_color[2] - 5)
            max_color = (label_color[0] + 5, label_color[1] + 5, label_color[2] + 5)
            new_mask[0, :, :, i] = cv2.inRange(mask[0], min_color, max_color)

        mask = new_mask
        mask = mask / 255
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1,
                    label_list = None, color_mask_dict = None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class,label_list,color_mask_dict)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for filename in sorted(os.listdir(test_path)):
        start_time = time.time()
        img = io.imread(os.path.join(test_path,filename),as_gray = False)
        img = img / 255
        img = cv2.resize(img,(target_size[1],target_size[0]))
        img = np.reshape(img,(1,)+img.shape)
        end_time = time.time()
        duration = end_time - start_time
        with open('logs/timeread.txt', "a") as log_file:
            log_file.write("Imread, s: " + str(duration) + "\n")
        yield img

#Self-made generator for reading images and masks from folder
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def labelVisualize(label_list, color_mask_dict, img, label_list_idxs=None):
    num_class = len(label_list)
    img_out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(num_class):
        if label_list_idxs is None:
            idx = i
            label_color = convert_str_to_rgb(color_mask_dict[label_list[idx]])
        else:
            idx = label_list_idxs[i]
            label_color = convert_str_to_rgb(color_mask_dict[label_list[i]])


        mask = img[:,:,idx]

        img_out[mask > 0.5] = np.array(label_color)

    return img_out

def saveResult(save_path,npyfile,flag_multi_class = False,label_list=None,color_mask_dict=None):
    for i,item in enumerate(npyfile):
        img = labelVisualize(label_list,color_mask_dict, item,save_path, i) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%04d_predict.png"%i),img)

