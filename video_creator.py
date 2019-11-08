import cv2
import glob
import numpy as np
import os

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/Projects/nkbvs_segmentation/dataset/joined.avi',fourcc, 10, (1280,384), True)
path_2 = 'D:/Projects/nkbvs_segmentation/dataset/results_multiclass_ct10_converted-500'
path_1 = 'D:/Datasets/NKBVS/mfti_data/2019-01-31-15-53-26_kia_velo_gps_time/stereo/left/image_raw/converted'
ext = 'png'

dir_list = [os.path.join(path_2, x) for x in os.listdir(path_2)]

if dir_list:
    # Создадим список из путей к файлам и дат их создания.
    date_list = [[x, os.path.getctime(x)] for x in dir_list]

    # Отсортируем список по дате создания в прямом порядке
    sort_date_list = sorted(date_list, key=lambda x: x[1], reverse=False)

list_1 = sorted(glob.glob(path_1+'\\*.'+ ext))
list_2 = sort_date_list
idx_max = 2000
for idx, img in enumerate(list_1):
    if idx < idx_max:
        print(img)
        frame_1 = cv2.imread(img)
        frame_1 = cv2.resize(frame_1, (640, 384))
        frame_2 = cv2.imread(list_2[idx][0])
        frame_2 = cv2.resize(frame_2, (640, 384))
        frame_3 = np.concatenate((frame_1, frame_2),axis=1)
        # write the flipped frame
        out.write(frame_3)
    else:
        break
print('End.')
cv2.destroyAllWindows()