import cv2
import glob
import numpy as np
import os
is_two = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
img_width_1 = 400
img_height = 500
img_width_2 = 400

#target_path = 'D:/Projects/nkbvs_segmentation/dataset/joined.avi'
#target_path = 'D:/Projects/nkbvs_localization/cross_cor/track_blue.avi'
#target_path = 'D:/Projects/nkbvs_localization/cross_cor/track_red.avi'
#target_path = 'D:/Projects/nkbvs_localization/cross_cor/track_green.avi'
target_path = 'D:/Projects/nkbvs_localization/cross_cor/track_compare.avi'

#path_1 = 'D:/Datasets/NKBVS/mfti_data/2019-01-31-15-53-26_kia_velo_gps_time/stereo/left/image_raw/converted'
#path_1 = 'D:/Projects/nkbvs_segmentation/dataset/results_2019-01-31-15-53-26_kia_velo_gps_time_with_dense'
#path_1= "D:/Projects/nkbvs_localization/cross_cor/cab_blue"
#path_1= "D:/Projects/nkbvs_localization/cross_cor/cab_red"
#path_1= "D:/Projects/nkbvs_localization/cross_cor/cab_green"
path_1 = "D:/Projects/nkbvs_localization/cross_cor/cv2.TM_CCORR_NORMED"
#ext = 'jpg'
ext = 'png'

if is_two:
    path_2 = "D:/Projects/nkbvs_localization/cross_cor/cv2.TM_CCORR_NORMED_blue"
    #path_2 = "D:/Projects/nkbvs_localization/cross_cor/cv2.TM_CCORR_NORMED_red"
    #path_2 = "D:/Projects/nkbvs_localization/cross_cor/cv2.TM_CCORR_NORMED_green"
    #path_2 = "D:/Projects/nkbvs_localization/cross_cor/cv2.TM_CCORR_NORMED"
    #path_2 = 'D:/Projects/nkbvs_segmentation/dataset/results_2019-01-31-15-53-26_kia_velo_gps_time_with_dense_softmax'

if is_two:
    out = cv2.VideoWriter(target_path,fourcc, 10, (img_width_1+img_width_2,img_height), True)
else:
    out = cv2.VideoWriter(target_path, fourcc, 10, (img_width_1, img_height), True)
idx_max = 2000


if is_two:
    dir_list = [os.path.join(path_2, x) for x in os.listdir(path_2)]

    if dir_list:
        # Создадим список из путей к файлам и дат их создания.
        date_list = [[x, os.path.getctime(x)] for x in dir_list]

        # Отсортируем список по дате создания в прямом порядке
        sort_date_list = sorted(date_list, key=lambda x: x[1], reverse=False)

list_1 = sorted(glob.glob(path_1+'\\*.'+ ext))
if is_two:
    list_2 = sort_date_list

for idx, img in enumerate(list_1):
    if idx < idx_max:
        print(img)
        frame_1 = cv2.imread(img)
        frame_1 = cv2.resize(frame_1, (img_width_1, img_height))
        if is_two:
            if (os.path.isfile(list_2[idx][0])):
                frame_2 = cv2.imread(list_2[idx][0])
                frame_2 = cv2.resize(frame_2, (img_width_2, img_height))
                frame_3 = np.concatenate((frame_1, frame_2),axis=1)
                # write the flipped frame
                out.write(frame_3)
        else:
            out.write(frame_1)
    else:
        break
print('End.')
cv2.destroyAllWindows()