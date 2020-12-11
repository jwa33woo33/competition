import os
import sys
import json
import glob
import subprocess
from collections import defaultdict
from PIL import Image

import numpy as np
from metric import generate_gt_json, evaluate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__=='__main__':
    # create_testdev()
    # create_submission_file(threshold=0.0)
    # validation()

    # pred_file = './result_jsons/temp.json'
    pred_file = './result_jsons/temp.json'
    # pred_file = './results.json'
    gt_file = './result_jsons/valid.json'


    with open (pred_file, 'r') as json_file:
        pred_file_json = json.load(json_file)
    
    print(len(pred_file_json))
    print('Loading annotations...')
    gt_annotations = COCO(gt_file)
    bbox_dets = gt_annotations.loadRes(pred_file)

    print('\nEvaluating BBoxes:')
    bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()

    for i in range(1, 8):
        print(f'\n Class: {i}')
        bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
        bbox_eval.params.catIds = i
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()


    gt_file = './result_jsons/ground_truth.json'
    pred_file = './t3_res_U0000000225.json'

    with open (pred_file, 'r') as json_file:
        submission = json.load(json_file)

    submit_cnt = 0
    for annot in submission['annotations']:
        for ann in annot['object']:
            submit_cnt += 1

    print(submit_cnt)


    generate_gt_json()
    f1 = evaluate(gt_file, pred_file)
    print(f'score: {f1:.5f}')
    # Also print coco evaluation