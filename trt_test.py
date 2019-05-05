#!/usr/bin/env python2
#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import sys, os
import common
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES



TRT_LOGGER = trt.Logger()

output_shapes_416 = [(1, 18, 13, 13), (1, 18, 26, 26)]
output_shapes_480 = [(1, 18, 15, 15), (1, 18, 30, 30)]
output_shapes_544 = [(1, 18, 17, 17), (1, 18, 34, 34)]
output_shapes_608 = [(1, 18, 19, 19), (1, 18, 38, 38)]
output_shapes_dic = {'416': output_shapes_416, '480': output_shapes_480, '544': output_shapes_544, '608': output_shapes_608}

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(engine_file_path=""):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    ENGINE_FILE_PATH = "ped3_416.trt"
    INPUT_LIST_FILE = './ped_list.txt'
    INPUT_SIZE = 416

    filenames = []
    with open(INPUT_LIST_FILE, 'r') as l:
        lines = l.readlines()
        for line in lines:
            filename = line.strip()
            filenames.append(filename)   
    input_resolution_yolov3_HW = (INPUT_SIZE, INPUT_SIZE)
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(8,34),  (14,60),  (23,94),  (39,149),  (87,291),  (187,472)],
                          "obj_threshold": 0.1,
                          "nms_threshold": 0.3,
                          "yolo_input_resolution": input_resolution_yolov3_HW}
    postprocessor = PostprocessYOLO(**postprocessor_args)
    output_shapes = output_shapes_dic[str(INPUT_SIZE)]
    with get_engine(ENGINE_FILE_PATH) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)       
        for filename in filenames:
            image_raw, image = preprocessor.process(filename)
            shape_orig_WH = image_raw.size
            trt_outputs = []
        
            # Do inference
            print('Running inference on image {}...'.format(filename))
            inputs[0].host = image
            c_time = 0
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t2 = time.time()
            c_time = t2-t1  
            print(c_time)
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
            if len(boxes) != 0:
                obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
            else:
                obj_detected_img = image_raw
            savename_0 = filename.split('/')[-1]
            savename = savename_0.split('.')[0]
            output_image_path = './images_results/' + savename + '_' + str(INPUT_SIZE) + '.png'
            obj_detected_img.save(output_image_path, 'PNG')
            print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    main()
