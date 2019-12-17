
import cv2
import numpy as np
import os

is_two = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
img_width_1 = 640
img_height = 384
img_width_2 = 640
fps = 10
idx_max = 2000 #First idx_max images will be recorded


target_path = 'dataset/results_unet_mct_softmax_2019-01-31-15-53-26_kia_velo_gps_time_stereo_left.avi'

path_1 = "/datasets/NKBVS_data/converted/2019-01-31-15-53-26_kia_velo_gps_time_stereo_left"


if is_two:
     path_2 = "/home/yuddim/github/cds-segmentation/dataset/results_unet_mct_softmax_2019-01-31-15-53-26_kia_velo_gps_time_stereo_left"

if is_two:
    out = cv2.VideoWriter(target_path, fourcc, fps, (img_width_1 + img_width_2, img_height), True)
else:
    out = cv2.VideoWriter(target_path, fourcc, fps, (img_width_1, img_height), True)

if is_two:
    dir_list = [os.path.join(path_2, x) for x in os.listdir(path_2)]

    if dir_list:
        # Создадим список из путей к файлам и дат их создания.
        date_list = [[x, os.path.getctime(x)] for x in dir_list]

        # Отсортируем список по дате создания в прямом порядке
        sort_date_list = sorted(date_list, key=lambda x: x[1], reverse=False)

list_1 = sorted([os.path.join(path_1, x) for x in os.listdir(path_1)])
if is_two:
    list_2 = sort_date_list

for idx, img in enumerate(list_1):
    if idx < idx_max:
        print(img)
        if os.path.isfile(img):
            frame_1 = cv2.imread(img)
            frame_1 = cv2.resize(frame_1, (img_width_1, img_height))
            if is_two:
                if (os.path.isfile(list_2[idx][0])):
                    frame_2 = cv2.imread(list_2[idx][0])
                    frame_2 = cv2.resize(frame_2, (img_width_2, img_height))
                    frame_3 = np.concatenate((frame_1, frame_2), axis=1)
                    # write the flipped frame
                    out.write(frame_3)
            else:
                out.write(frame_1)
    else:
        break
print('End.')
#cv2.destroyAllWindows()



