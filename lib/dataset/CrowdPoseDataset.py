# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

from crowdposetools.cocoeval import COCOeval
from utils import zipreader
from utils.rescore import CrowdRescoreEval
import pdb
logger = logging.getLogger(__name__)

MODALITY_MAP = {"sharp": 'gt_processed', "blur2blur": "blur2blur_processed", 
                             "blur": "blur_processed", "blurred": "blurred_processed", "event": "event_voxel"}


class CrowdPoseDataset(Dataset):
    def __init__(self, cfg, dataset):
        from crowdposetools.coco import COCO
        self.root = cfg.DATASET.ROOT
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints+1
        self.domain = "source" if len(cfg.DATASET.DOMAIN_MODALITY.SOURCE) else "target"
        self.modality = {
            'source': 
                {'image': next((k for k in ['sharp', 'blur', 'blurred', 'blur2blur'] if k in cfg.DATASET.DOMAIN_MODALITY.SOURCE), None),
                'event': next((k for k in ['event'] if k in cfg.DATASET.DOMAIN_MODALITY.SOURCE), None)
                },
            'target':
                {'image': next((k for k in ['sharp', 'blur', 'blurred', 'blur2blur'] if k in cfg.DATASET.DOMAIN_MODALITY.TARGET), None),
                'event': next((k for k in ['event'] if k in cfg.DATASET.DOMAIN_MODALITY.TARGET), None)
                }
            }
        json_path = os.path.join(
            self.root,
            f'test_dir_{self.domain}.json'
        )
        # self.coco = COCO(self._get_anno_file_name())
        self.coco = COCO(json_path)
        self.ids = list(self.coco.imgs.keys())
        self.da_setting = cfg.DATASET.DA_SETTING
        self.dir_name = self.root.split('/Annotation')[0]
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )


            
    def _get_anno_file_name(self):
        # example: root/json/crowdpose_{train,val,test}.json
        dataset = 'trainval' if 'rescore' in self.dataset else self.dataset
        return os.path.join(
            self.root,
            'json',
            'crowdpose_{}.json'.format(
                dataset
            )
        )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        if self.data_format == 'zip':
            return images_dir + '.zip@' + file_name
        else:
            return os.path.join(images_dir, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info['file_name']
        image_path = os.path.join(self.dir_name, file_name)
        object_metas = target
        gt_joints = self.get_joints_from_json(object_metas)
        event_path = image_path.replace("png", "npz").replace("gt_processed" if self.modality == 'source' else "blur_processed", "event_voxel")
        f_flow_path = event_path.replace("event_voxel/", "flow/f_")
        b_flow_path = event_path.replace("event_voxel/", "flow/b_")        

        voxel = np.load(event_path)["data"]
        f_flow = np.load(f_flow_path)['flow']
        b_flow = np.load(b_flow_path)['flow']
        
        img = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flow = np.concatenate([f_flow, b_flow], axis = 0)
        
        if 'train' in self.dataset:
            return img, [obj for obj in target], image_info, voxel
        else:
            return img, voxel, image_info, flow, gt_joints

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str
    
    def merge_half_voxel(self, front, back):
        c, h, w = front.shape
        merged = np.zeros((2*c-1,h,w))
        merged[:c] += front
        merged[-c:] += back
        return merged

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def get_joints_from_json(self, anno):
        num_people = len(anno)
        joints = np.zeros((num_people, self.num_joints_with_center, 3))
        for i in range(num_people):
            key_points = anno[i]['keypoints']
            if len(key_points) == 0:
                continue
            
            for k_ipt in range(self.num_joints):
                x_coord = float(anno[i]['keypoints'][3*k_ipt])
                y_coord = float(anno[i]['keypoints'][3*k_ipt+1])
                
                joints[i, k_ipt, 0] = x_coord
                joints[i, k_ipt, 1] = y_coord
                joints[i, k_ipt, 2] = (anno[i]['keypoints'][3*k_ipt] !=0)

            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = anno[i]['num_keypoints']
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints
    
    def get_joints(self, anno):
        num_people = len(anno)
        joints = np.zeros((num_people, self.num_joints_with_center, 3))
        pdb.set_trace()
        for i in range(num_people):
            key_points = anno[i]['keypoints'][0]
            if len(key_points) == 0:
                continue
            
            for k_ipt in range(len(key_points)):
                x_coord = float(anno[i]['keypoints'][0][k_ipt][0])
                y_coord = float(anno[i]['keypoints'][0][k_ipt][1])
                
                joints[i, k_ipt, 0] = x_coord
                joints[i, k_ipt, 1] = y_coord
                
                
                if x_coord != 0 or y_coord != 0:
                    joints[i, k_ipt, 2] = 1
                else:
                    joints[i, k_ipt, 2] = 0
            # joints[i, :self.num_joints, :3] = \
            #     np.array(obj['keypoints']).reshape([-1, 3])


            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints
    def evaluate(self, cfg, preds, scores, output_dir, tag,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % (self.dataset+tag))

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                file_name = file_name.split('/')[-1]
                kpts[file_name].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        # 'image': int(file_name.split('.')[0]),
                        'image': img_id,
                        'area': area
                    }
                )
        self.save_pseudo_label(preds, scores, res_file.replace("regression", "pseudolabel"))

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])
        
        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )
        # CrowdPose `test` set has annotation.
        info_str = self._do_python_keypoint_eval(
            res_file, res_folder
        )
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']
    

    def save_pseudo_label(self, preds, scores, output_dir):
        res_file = output_dir.replace("train", "train_with_pseudo")
        
        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        res = {
            "categories": [{
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "head",
                    "neck"
                ]
            }],
            "images": [],
            "annotations": []
        }
        
        ann_id = 0
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            image_entry = self.coco.loadImgs(img_id)
            res['images']+=(image_entry)
            for idx_kpt, kpt in enumerate(_kpts):
                kep_mask = kpt[:,2]>0
                kpt[:, 2] =1
                kep['keypoints'] = kpt.reshape(-1).tolist()
                if kep_mask.sum() == 0:
                    
                    kep['bbox'] = [0,0,0,0]
                else:
                    left_top = np.amin(kpt[kep_mask],axis=0)[0:2].tolist()
                    right_bottom = np.amax(kpt[kep_mask],axis=0)[0:2].tolist()
                    kep['bbox'] = [left_top[0], left_top[1], right_bottom[0]-left_top[0], right_bottom[1]-left_top[1]]
                kep['image_id'] = img_id
                kep['category_id'] =1
                kep["iscrowd"] =0
                kep['score'] = scores[idx][idx_kpt]
                kep['num_keypoints'] = kep_mask.sum()
                kep['id'] = ann_id
                res['annotations'].append(kep)
                ann_id +=1
        
        with open(res_file, 'w') as f:
            json.dump(res, f, indent=4)
        

    def evaluate_and_save_from_file(self):
        res_file = 'output/crowd_pose_kpt/hrnet_dekr/<config_name>/results/keypoints_testregression_results.json'
        with open(res_file, 'r') as f:
            data = json.load(f)
        
        res = {
            "categories": [{
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "head",
                    "neck"
                ]
            }],
            "images": [],
            "annotations": []
        }
        ann_id = 0
        for image_id in list(self.coco.imgs.keys()):
            res['images'].append(self.coco.loadImgs(image_id)[0])
        pdb.set_trace()
        for idx, _kpts in enumerate(data):
            kep = {}
            keypoint = _kpts['keypoints']
            kpt = np.array(keypoint).reshape(14,3)
            kep_mask = kpt[:,2]>0.3
            kpt[:,2] = 0
            kpt[kep_mask, 2] =1
            kep['keypoints'] = kpt.reshape(-1).tolist()
            if kep_mask.sum() == 0:
                kep['bbox'] = [0,0,0,0]
            else:
                left_top = np.amin(kpt[kep_mask],axis=0)[0:2].tolist()
                right_bottom = np.amax(kpt[kep_mask],axis=0)[0:2].tolist()
                kep['bbox'] = [left_top[0], left_top[1], right_bottom[0]-left_top[0], right_bottom[1]-left_top[1]]
            kep['image_id'] = _kpts['image_id']
            kep['category_id'] =1
            kep["iscrowd"] =0
            kep['score'] = _kpts['score']
            kep['num_keypoints'] = kep_mask.sum()
            kep['id'] = ann_id
            res['annotations'].append(kep)
            ann_id +=1
        with open(target_file, 'w') as f:
            json.dump(res, f, indent=4)
        

    def evaluate_from_file(self,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_file = 'output/crowd_pose_kpt/hrnet_dekr/<config_name>/results/keypoints_testregression_results.json'        

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        

        info_str = self._do_python_keypoint_eval(
            res_file, None
        )
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']
    
    def evaluate_save(self, cfg, preds, scores, output_dir, 
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(*output_dir.split('/')[:-1])
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        # res_file = os.path.join(
        #     res_folder, 'keypoints_%s_results.json' % (self.dataset+tag))

        res_file = output_dir
        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        
        for idx, _kpts in enumerate(preds):
            img_id = self.target_ids[int(idx/100)]+(idx%100)
            # file_name = self.coco.loadImgs(img_id)[0]['file_name']
            file_name = output_dir.split('/')[-1]
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                file_name = file_name.split('/')[-1]
                kpts[file_name].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        # 'image': int(file_name.split('.')[0]),
                        'image': img_id,
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )
        return 0
        # # CrowdPose `test` set has annotation.
        # info_str = self._do_python_keypoint_eval(
        #     res_file, res_folder
        # )
        # name_value = OrderedDict(info_str)
        # return name_value, name_value['AP']

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 14

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        #coco_eval.params.iouThrs = np.array([0.001])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5',
                       'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
        stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[stats_index[ind]]))
            # info_str.append(coco_eval.stats[ind])

        return info_str


class CrowdPoseRescoreDataset(CrowdPoseDataset):
    def __init__(self, cfg, dataset):
        CrowdPoseDataset.__init__(self, cfg, dataset)

    def evaluate(self, cfg, preds, scores, output_dir,
                 *args, **kwargs):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.dataset)
        pdb.set_trace()
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * \
                    (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)

                kpts[int(file_name.split('.')[0])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'image': int(file_name.split('.')[0]),
                        'area': area
                    }
                )

        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        self._do_python_keypoint_eval(
            cfg.RESCORE.DATA_FILE, res_file, res_folder
        )

    def _do_python_keypoint_eval(self, data_file, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = CrowdRescoreEval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.dumpdataset(data_file)