#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
from statistics import mean 

import torch

from yolox.data.datasets import DAIS_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=DAIS_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=DAIS_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class DAISEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
        mag_tape = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.mag_tape = mag_tape
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        ids = []
        data_list = []
        output_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        # YOLinO evaluation
        if self.mag_tape:
            for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
            ):
                with torch.no_grad():
                    imgs = imgs.type(tensor_type)

                    # skip the last iters since batchsize might be not enough for batch inference
                    is_time_record = cur_iter < len(self.dataloader) - 1
                    if is_time_record:
                        start = time.time()

                    outputs = model(imgs)

                    if is_time_record:
                        infer_end = time_synchronized()
                        inference_time += infer_end - start

                    # TODO: Add postprocessing of the outputs like DBSCAN and NMS later

                # TEST
                output_list.extend(outputs)

            statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
            if distributed:
                # different process/device might have different speed,
                # to make sure the process will not be stucked, sync func is used here.
                synchronize()
                output_list = gather(output_list, dst=0)
                output_list = list(itertools.chain(*output_list))
                torch.distributed.reduce(statistics, dst=0)

            eval_results = self.evaluate_yolino_prediction(output_list, statistics)
            synchronize()

            if return_outputs:
                return eval_results, output_list
            return eval_results

                    
        # YOLOX evaluation
        else:
            for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
            ):
                with torch.no_grad():
                    imgs = imgs.type(tensor_type)

                    # skip the last iters since batchsize might be not enough for batch inference
                    is_time_record = cur_iter < len(self.dataloader) - 1
                    if is_time_record:
                        start = time.time()

                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    if is_time_record:
                        infer_end = time_synchronized()
                        inference_time += infer_end - start

                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                    if is_time_record:
                        nms_end = time_synchronized()
                        nms_time += nms_end - infer_end

                data_list_elem, image_wise_data = self.convert_to_coco_format(
                    outputs, info_imgs, ids, return_outputs=True)
                data_list.extend(data_list_elem)
                output_data.update(image_wise_data)

            statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
            if distributed:
                # different process/device might have different speed,
                # to make sure the process will not be stucked, sync func is used here.
                synchronize()
                data_list = gather(data_list, dst=0)
                output_data = gather(output_data, dst=0)
                data_list = list(itertools.chain(*data_list))
                output_data = dict(ChainMap(*output_data))
                torch.distributed.reduce(statistics, dst=0)

            eval_results = self.evaluate_prediction(data_list, statistics)
            synchronize()

            if return_outputs:
                return eval_results, output_data
            return eval_results

    def evaluate_yolino_prediction(self, output_list, statistics, t_conf=0.5, t_cell=0.1):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        batch_size = self.dataloader.batch_size
        annotations = self.dataloader.dataset.annotations

        # Extract the ground truths from the annotations
        # print(ground_truth[0] for ground_truth in annotations[:2])
        ground_truths = [torch.from_numpy(ground_truth[0]).permute(1, 2, 0) for ground_truth in annotations] # Mozda kje treba da dodades edna nula kaj permute i drugite indeksi da gi zgolemis za 1

        metrics_yolino = {}
        if len(ground_truths) == len(output_list):
            precision_list = []
            recall_list = []
            f1_score_list = []

            precision_cell_list = []
            recall_cell_list = []
            f1_score_cell_list = []

            for i in range(len(output_list)):
                prediction = output_list[i]
                # print(f"Prediction shape: {prediction.shape}")
                ground_truth = ground_truths[i]
                # print(f"Ground truth shape: {ground_truth.shape}")

                # print(f"Ground truth: {ground_truth}, prediction: {prediction}")
                
                # Reshape ground truth as prediction
                ground_truth = ground_truth.reshape_as(prediction)
                # print(f"Ground truth shape: {ground_truth.shape}")

                """ CHECK ONLY DISTANCES BETWEEN START AND END POINTS OF GT AND PREDICTION """
                coords_gt = ground_truth[:, :, :4].float().to("cpu")
                confs_gt = ground_truth[:, :, -1].float().to("cpu")

                coords_pred = prediction[:, :, :4].float().to("cpu")
                confs_pred = prediction[:, :, -1].float().to("cpu")

                num_ground_truths = (confs_gt > 0).sum(dim=0)
                # print(f"Number of ground truths: {num_ground_truths}")

                num_predictions = (confs_pred > t_conf).sum(dim=0)
                # print(f"Number of predictions: {num_predictions}")

                # HIGH LEVEL METRICS
                # True positives
                true_positives = torch.logical_and(confs_gt > 0, confs_pred > t_conf).sum(dim=0)
                # print(f"True positives: {true_positives}")

                # False positives
                false_positives = torch.logical_and(confs_gt < 1, confs_pred > t_conf).sum(dim=0)
                # print(f"False positives: {false_positives}")

                # True negatives
                true_negatives = torch.logical_and(confs_gt < 1, confs_pred < t_conf).sum(dim=0)
                # print(f"True negatives: {true_negatives}")

                # False negatives
                false_negatives = torch.logical_and(confs_gt > 0, confs_pred < t_conf).sum(dim=0)
                # print(f"False negatives: {false_negatives}")

                # Precision
                precision = true_positives / (num_predictions + 1e-15)
                # print(f"Precision: {precision.item()}")

                # Recall
                recall = true_positives / (num_ground_truths + 1e-15)
                # print(f"Recall: {recall}")

                # F1
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)
                # print(f"F1 score: {f1_score}")

                # Append the lists for every image
                precision_list.append(precision.item())
                recall_list.append(recall.item())
                f1_score_list.append(f1_score.item())

                # CELL LEVEL METRICS
                calculate_distances = torch.logical_and(confs_gt > 0, confs_pred > t_conf)
                check_distances = torch.logical_and(torch.linalg.norm(coords_gt[:, :, :2] - coords_pred[:, :, :2], dim=-1) < t_cell, 
                                                    torch.linalg.norm(coords_gt[:, :, 2:] - coords_pred[:, :, 2:], dim=-1) < t_cell)

                true_positives_cell = torch.logical_and(calculate_distances, check_distances).sum(dim=0)

                # print(f"Cell based true positives: {true_positives_cell}")

                # Cell based precision
                precision_cell = true_positives_cell / (num_predictions + 1e-15)
                # print(f"Cell based precision: {precision_cell}")

                # Cell based recall
                recall_cell = true_positives_cell / (num_ground_truths + 1e-15)
                # print(f"Cell based recall: {recall_cell}")

                # Cell based F1 score
                f1_score_cell = 2 * (precision_cell * recall_cell) / (precision_cell + recall_cell + 1e-15)

                # Append the lists for every image
                precision_cell_list.append(precision_cell.item())
                recall_cell_list.append(recall_cell.item())
                f1_score_cell_list.append(f1_score_cell.item())

            metrics_yolino.update({"precision": mean(precision_list),
                                  "recall": mean(recall_list),
                                  "f1_score": mean(f1_score_list),
                                  
                                  "cell_based_precision": mean(precision_cell_list),
                                  "cell_based_recall": mean(recall_cell_list),
                                  "cell_based_f1_score": mean(f1_score_cell_list)})
            
        return metrics_yolino, info
    
    # Integrate afterwards
    def interpolate_line_segment(self, point1, point2):
        # Calculate the number of steps based on the distance
        distance = np.linalg.norm(np.array(point2) - np.array(point1))
        print(distance)
        
        min_points = 2
        density_factor = 10.0

        # Generate interpolated points
        num_points = max(int(np.ceil(distance * density_factor)), min_points) + 1
        print(num_points)

        # (x2 > x1 and y2 > y1) or (x2 > x1 and y2 < y1)
        if (point2[0] > point1[0] and point2[1] > point1[1]) or (point2[0] > point1[0] and point2[1] < point1[1]):
            x_values = np.linspace(point1[0], point2[0], num_points)
            y_values = np.linspace(point1[1], point2[1], num_points)
            interpolated_points = list(zip(x_values, y_values))

        # (x2 < x1 and y2 < y1) or (x2 < x1 and y2 > y1)
        elif (point2[0] < point1[0] and point2[1] <  point1[1]) or (point2[0] < point1[0] and point2[1] > point1[1]):
            x_values = np.linspace(point2[0], point1[0], num_points)
            y_values = np.linspace(point2[1], point1[1], num_points)
            interpolated_points = list(zip(x_values, y_values))
            interpolated_points.reverse()

        return interpolated_points
    
    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
