import os
import cv2
import glob
import h5py
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import YOU2ME, TCMR_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import get_bbox_from_kp2d
from lib.data_utils._feature_extractor import extract_features

from lib.data_utils._occ_utils import load_occluders

cmu_train_list = ['1-catch1', '10-hand1', '11-hand2', '12-hand3', 
                '13-sports1', '14-sports2', 
                '2-catch2', '3-catch3', 
                '4-convo1', '5-convo2', '6-convo3', '7-convo4', '8-convo5', '9-convo6']

kinect_train_list = ['catch36', 'catch37', 'catch39', 'catch40', 'catch41', 'catch42', 'catch55', 
'convo43', 'convo46', 'convo47', 'convo53', 'convo54', 'convo59', 
'patty1', 'patty2', 'patty26', 'patty27', 'patty28', 'patty30', 'patty31', 'patty32', 'patty34', 'patty35', 'patty5', 
'sport56', 'sport57', 'sport58']

def read_body3DScene(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['bodies']
    body_0_joints = people[0]["joints19"]
    body_1_joints = people[1]["joints19"]
    return body_0_joints, body_1_joints

def read_openpose(json_file): # gt_part
        # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        # TODO scale of person
        #scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                # TODO try just get the closest one
                dist_conf[i] = np.mean(np.sqrt(np.sum(op_conf12*op_keyp12)**2, axis=1)) # *(op_keyp12 - gt_part[:12, :2]
        # closest match
        # There maybe many matches and here we only wnat the cloest
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        thresh = 0
        # dataset-specific thresholding based on pixel size of person
        #if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
        #    keyp25 = np.zeros([25,3])
        #else:
        keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def read_train_data(dataset_path, data_type, debug=False):
    h, w = 227, 227
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'egojoints3D':[],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    # occluders = load_occluders('./data/VOC2012')

    model = spin.get_pretrained_hmr()

    # training data
    # two types of data
    # read cmu first
    if data_type == 'cmu':
        for seq_num, vid_i in enumerate(cmu_train_list):
            print("vid_i: ", vid_i)
            print("seq_num: ", seq_num)
            imgs_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'synchronized',
                                        'frames')
            pattern = os.path.join(imgs_path, '*.jpg')
            img_list = sorted(glob.glob(pattern))
            print('imgs_path',imgs_path)

            openpose_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'features',
                                        'openpose',
                                        'output_json')
            gt_skeletons_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'synchronized',
                                        'gt-skeletons')

            for i, img_i in tqdm_enumerate(img_list):
                img_name = img_i.split('/')[-1]
                openpose_name = img_name.split('.')[0] + '_keypoints.json'
                openpose_i = os.path.join(openpose_path,openpose_name)
                # try read you2me keypiont
                joints_2d_raw = read_openpose(openpose_i).reshape(1, 25, 3)
                # joints_2d_raw[:,:,2::3] = len(joints_2d_raw[:,:,2::3][2::3])*[1] # set confidence to 1
                # key2djnts[2::3] = len(key2djnts[2::3])*[1]
                # print('joints_2d',joints_2d_raw)
                joints_2d = convert_kps(joints_2d_raw, "you2me2d",  "spin").reshape((-1,3))
                # print('joints_2d',np.shape(joints_2d))
                # TODO what is the difference between openpose joint keypoint and convert one
                joints_3d_name = 'body3DScene_' + img_name.split('x')[-1].split('.')[0]
                interact_joints_3d, ego_1_joints_3d =  os.path.join(gt_skeletons_path,joints_3d_name)
                joints_3d_raw = np.reshape(interact_joints_3d, (1, 19, 4)) / 1000 # TODO why divide 1000
                joints_3d_raw = joints_3d_raw[:,:,:3]

                # print('joints_3d_name',joints_3d_name)
                joints_3d = convert_kps(joints_3d_raw, "you2me_cmu_3d", "spin").reshape((-1,3))
                

            




    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    # product = product(user_list, seq_list, vid_list)
    # user_i, seq_i, vid_i = product[process_body_0_joints_3did]

    for user_i in user_list:
        print("Subject: ", user_i)
        for seq_i in seq_list:
            print("seq_i: ", seq_i)
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            for j, vid_i in enumerate(vid_list):
                print("vid_i: ", vid_i)
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'video_' + str(vid_i))
                # per frame
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                vid_used_frames = []
                vid_used_joints = []
                vid_used_bbox = []
                vid_segments = []
                vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i) + "_vid" + str(vid_i) + "_seg0"
                for i, img_i in tqdm_enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    joints_2d_raw = np.reshape(annot2[vid_i][0][i], (1, 28, 2))
                    joints_2d_raw= np.append(joints_2d_raw, np.ones((1,28,1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "mpii3d",  "spin").reshape((-1,3))

                    joints_3d_raw = np.reshape(annot3[vid_i][0][i], (1, 28, 3)) / 1000
                    joints_3d = convert_kps(joints_3d_raw, "mpii3d", "spin").reshape((-1,3))

                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    # check that all joints are visible
                    x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                    y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < joints_2d.shape[0]:
                        vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                                          str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                        continue


                    visualize = False
                    if visualize == True and i > 500:
                        import matplotlib.pyplot as plt

                        frame = cv2.cvtColor(cv2.imread(img_i), cv2.COLOR_BGR2RGB)

                        for k in range(49):
                            kp = joints_2d[k]

                            frame = cv2.circle(
                                frame.copy(),
                                (int(kp[0]), int(kp[1])),
                                thickness=3,
                                color=(255, 0, 0),
                                radius=5,
                            )

                            cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                        (0, 255, 0),
                                        thickness=3)

                        cv2.imshow('vis', frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(img_name.split(".")[0])
                    dataset['img_name'].append(img_i)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_segments.append(vid_uniq_id)
                    vid_used_frames.append(img_i)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

                vid_segments= np.array(vid_segments)
                ids = np.zeros((len(set(vid_segments))+1))
                ids[-1] = len(vid_used_frames) + 1
                if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                    ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

                for i in tqdm(range(len(set(vid_segments)))):
                    features = extract_features(model, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                                vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                                kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                                dataset='spin', debug=False, scale=1.0)
                    dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


def read_test_data(dataset_path):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        "valid_i": []
    }

    model = spin.get_pretrained_hmr()

    user_list = range(1, 7)

    for user_i in user_list:
        print('Subject', user_i)
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])

        vid_used_frames = []
        vid_used_joints = []
        vid_used_bbox = []
        vid_segments = []
        vid_uniq_id = "subj" + str(user_i) + "_seg0"

        for frame_i, valid_i in tqdm(enumerate(valid)):
            img_i = os.path.join('mpi_inf_3dhp_test_set',
                                    'TS' + str(user_i),
                                    'imageSequence',
                                    'img_' + str(frame_i + 1).zfill(6) + '.jpg')

            joints_2d_raw = np.expand_dims(annot2[frame_i, 0, :, :], axis = 0)
            joints_2d_raw = np.append(joints_2d_raw, np.ones((1, 17, 1)), axis=2)


            joints_2d = convert_kps(joints_2d_raw, src="mpii3d_test", dst="spin").reshape((-1, 3))

            visualize = False
            if visualize == True:
                frame = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, img_i)), cv2.COLOR_BGR2RGB)

                for k in range(49):
                    kp = joints_2d[k]

                    frame = cv2.circle(
                        frame.copy(),
                        (int(kp[0]), int(kp[1])),
                        thickness=3,
                        color=(255, 0, 0),
                        radius=5,
                    )

                    cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),
                                thickness=3)

                cv2.imshow(f'frame:{frame_i}', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)


            joints_3d_raw = np.reshape(annot3[frame_i, 0, :, :], (1, 17, 3)) / 1000
            joints_3d = convert_kps(joints_3d_raw, "mpii3d_test", "spin").reshape((-1, 3))
            joints_3d = joints_3d - joints_3d[39] # substract pelvis zero is the root for test

            bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_i)
            I = cv2.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
            y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)

            if np.sum(ok_pts) < joints_2d.shape[0]:
                vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1]) + "_seg" + \
                              str(int(dataset['vid_name'][-1].split("_")[-1][3:]) + 1)
                continue

            dataset['vid_name'].append(vid_uniq_id)
            dataset['frame_id'].append(img_file.split("/")[-1].split(".")[0])
            dataset['img_name'].append(img_file)
            dataset['joints2D'].append(joints_2d)
            dataset['joints3D'].append(joints_3d)
            dataset['bbox'].append(bbox)
            dataset['valid_i'].append(valid_i)

            vid_segments.append(vid_uniq_id)
            vid_used_frames.append(img_file)
            vid_used_joints.append(joints_2d)
            vid_used_bbox.append(bbox)

        vid_segments = np.array(vid_segments)
        ids = np.zeros((len(set(vid_segments)) + 1))
        ids[-1] = len(vid_used_frames) + 1
        if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
            ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

        for i in tqdm(range(len(set(vid_segments)))):
            features = extract_features(model, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i + 1])],
                                        vid_used_bbox[int(ids[i]):int(ids[i + 1])],
                                        kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i + 1])],
                                        dataset='spin', debug=False, scale=1.2)  # 1.0 for mpii3d_train_scale1_db.pt
            dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/you2me')
    parser.add_argument('--data_type', type=str,choices=['cmu','kinect'],help='two kinds of data', default='kinect')
    parser.add_argument('--debug', type=bool, help='debug model', default=False)
    
    args = parser.parse_args()

    dataset = read_train_data(args.dir,args.data_type, args.debug)
    joblib.dump(dataset, osp.join(TCMR_DB_DIR, 'you2me_train_db.pt'))

    # dataset = read_test_data(args.dir,args.data_type,args.debug)
    # joblib.dump(dataset, osp.join(TCMR_DB_DIR, 'you2me_val_db.pt'))





