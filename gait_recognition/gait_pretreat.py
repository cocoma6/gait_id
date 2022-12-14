import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from warnings import warn
import shutil

OUTPUT_PATH = './data/output/' # set the output path
POST_PATH = './data/pretreat/' # set the output path after pretreatment
INPUT_PATH = './data/walking/' # set the input path
LOG_PATH = './pretreatment.log'

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

# resize the silhoette as 64 * 64
T_H = 64
T_W = 64

# def display_video(path):
#     cap = cv2.VideoCapture(path)
#     f_list = list()
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             f_list.append(frame)
#             # Press Q on keyboard to  exit
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else: 
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     return f_list

def image_enhancement(frame):
    err_kernel = np.ones((3, 3), np.uint8)
    dil_kernel = np.ones((3, 3), np.uint8)

    img_erosion = cv2.erode(frame, err_kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, dil_kernel, iterations=1)
    return img_dilation

def set_new_repo(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("new folder: %s" %path)
    else:
        print("update folder: %s" %path)

def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')

def cut_img(img, seq_info, frame_name, pid):
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (seq_info, frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')

def check_new_view(previous_idx, cur):
    if len(previous_idx) > 5:
        sum = cur - previous_idx[-1]
        for i in range(1,4):
            sum += previous_idx[-i] - previous_idx[-i-1] 
        if sum > 40:
            return True
    return False
    
'''
source: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
'''
def background_subtract(avi_name, sub_id, cond_id, pid):
    in_path = os.path.join(INPUT_PATH, avi_name)
    full_path = os.path.join(OUTPUT_PATH, avi_name)
    set_new_repo(full_path)

    p_full_path = os.path.join(POST_PATH, sub_id, cond_id).replace("\\", "/")
    set_new_repo(p_full_path)

    backSub = cv2.createBackgroundSubtractorKNN()
    backSub. setHistory(600)
    backSub.setShadowThreshold(0.6)
    backSub.setDist2Threshold(800)
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(in_path))
    if not capture.isOpened():
        print('Unable to open: ' + in_path)
        exit(0)

    fnum = 0
    view = 0
    idx_list = list()
    while True:
        ret, frame = capture.read()
        idx = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame is None:
            break
        
        fgMask1 = backSub.apply(frame) # background subtraction
        _, fgMask2 = cv2.threshold(fgMask1, 200, 255, type=0) # get rid of shadow
        fgMask3 = cv2.medianBlur(fgMask2, 3) # median filter
        fgMask = image_enhancement(fgMask3) # image morphology

        # A silhouette contains too little white pixels might be not valid for identification
        if fgMask.sum() > 255*500 and fgMask.sum() < 255*120*150:
            frame_name = "silho_%d.jpg" %idx
            save_path = os.path.join(full_path, frame_name)
            cv2.imwrite(save_path, fgMask)
            idx_list.append(idx)

            if check_new_view(idx_list, idx):
                view += 1
                fnum = 0
            pretreat_path = os.path.join(p_full_path, str(view)).replace("\\", "/")
            if not os.path.exists(pretreat_path):
                os.mkdir(pretreat_path)
            cutMask = cut_img(fgMask, cond_id, fnum, pid)
            frame_name = "processed_%d.jpg" %fnum
            post_save_path = os.path.join(pretreat_path, frame_name)
            cv2.imwrite(post_save_path, cutMask)
            fnum += 1
            pid += 1

            # plot the result
            # fig = plt.figure(figsize=(14, 4))
            # ax1 = fig.add_subplot(1,3,1)  
            # ax1.imshow(frame)
            # ax1.set_title("frame: %d"%idx)
            # ax2 = fig.add_subplot(1,3,2)  
            # ax2.imshow(fgMask1, cmap='gray')
            # ax2.set_title("background substraction")
            # ax3 = fig.add_subplot(1,3,3)
            # ax3.imshow(fgMask, cmap='gray')
            # ax3.set_title("silhouette")
            # plt.savefig("%s/result_%d.jpg" %(full_path, idx))
            # plt.show()
            
        # else:
        #     print("Working on frame %d, no silhouette" %idx)
        
# main
if __name__ == '__main__':
    pid = 0
    os.chdir('./gait_recognition/')

    # Walk the input path
    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    for _avi in id_list:
        print("working on:", _avi)
        tmp = str.split(_avi,'.')
        _id = tmp[0]
        attr_list = str.split(_id, '_')
        sub_id = attr_list[0]
        cond_id = attr_list[2]
        print(_avi, sub_id, cond_id)
        background_subtract(_avi, sub_id, cond_id, pid)

    for _label in sorted(list(os.listdir(POST_PATH))): # _label: people id
        label_path = os.path.join(POST_PATH, _label).replace("\\", "/")
        for _seq_type in sorted(list(os.listdir(label_path))): # _seq_type: walking condtition
            seq_type_path = os.path.join(label_path, _seq_type).replace("\\", "/")
            for _view in sorted(list(os.listdir(seq_type_path))): # _view: only keep the effective view
                _seq_dir = os.path.join(seq_type_path, _view).replace("\\", "/")
                seqs = os.listdir(_seq_dir)
                if len(seqs) < 30:
                    shutil.rmtree(_seq_dir)
