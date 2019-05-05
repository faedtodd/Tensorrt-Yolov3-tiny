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

import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

# from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import wget

TRT_LOGGER = trt.Logger()

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

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 16
            builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def download_file(path, link, checksum_reference=None):
    if not os.path.exists(path):
        print('downloading')
        wget.download(link, path)
        print()
    if checksum_reference is not None:
        raise ValueError('error')
    return path
def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'ped3_416_b16.onnx'
    engine_file_path = "ped3_416_b16.trt"
    
    f_name1 = '1'
    f_name2 = '2'
    f_name3 = '3'
    f_name4 = '4'
    f_name5 = '5'
    f_name6 = '6'
    f_name7 = '7'
    f_name8 = '8'
    input_image_path1 = './image_PNG/'+ f_name1 + '.jpg'
    input_image_path2 = './image_PNG/'+ f_name2 + '.jpg'
    input_image_path3 = './image_PNG/'+ f_name3 + '.jpg'
    input_image_path4 = './image_PNG/'+ f_name4 + '.jpg'
    input_image_path5 = './image_PNG/'+ f_name5 + '.jpg'
    input_image_path6 = './image_PNG/'+ f_name6 + '.jpg'
    input_image_path7 = './image_PNG/'+ f_name7 + '.jpg'
    input_image_path8 = './image_PNG/'+ f_name8 + '.jpg' 

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (416, 416)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw1, image1 = preprocessor.process(input_image_path1)
    image_raw2, image2 = preprocessor.process(input_image_path2)
    image_raw3, image3 = preprocessor.process(input_image_path3)
    image_raw4, image4 = preprocessor.process(input_image_path4)
    image_raw5, image5 = preprocessor.process(input_image_path5)
    image_raw6, image6 = preprocessor.process(input_image_path6)
    image_raw7, image7 = preprocessor.process(input_image_path7)
    image_raw8, image8 = preprocessor.process(input_image_path8)
    raw = []
    raw.append(image_raw1)
    raw.append(image_raw2)
    raw.append(image_raw3)
    raw.append(image_raw4)
    raw.append(image_raw5)
    raw.append(image_raw6)
    raw.append(image_raw7)
    raw.append(image_raw8)
    raw.append(image_raw1)
    raw.append(image_raw2)
    raw.append(image_raw3)
    raw.append(image_raw4)
    raw.append(image_raw5)
    raw.append(image_raw6)
    raw.append(image_raw7)
    raw.append(image_raw8)
    raw.append(image_raw1)
    raw.append(image_raw2)
    raw.append(image_raw3)
    raw.append(image_raw4)
    raw.append(image_raw5)
    raw.append(image_raw6)
    raw.append(image_raw7)
    raw.append(image_raw8)
    raw.append(image_raw1)
    raw.append(image_raw2)
    raw.append(image_raw3)
    raw.append(image_raw4)
    raw.append(image_raw5)
    raw.append(image_raw6)
    raw.append(image_raw7)
    raw.append(image_raw8)

    image_all = np.concatenate((image1, image2, image3, image4, image5, image6, image7, image8, image1, image2, image3, image4, image5, image6, image7, image8, image1, image2, image3, image4, image5, image6, image7, image8, image1, image2, image3, image4, image5, image6, image7, image8), axis=0)
    print(image1.shape)
    print(image_all.shape)
    # print(image)
    # Store the shape of the original input image in WH format, we will need it for later
    

    # Output shapes expected by the post-processor
    output_shapes = [(16, 18, 13, 13), (16, 18, 26, 26)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path1))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image_all[:16]
        t1 = time.time()
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        t2 = time.time()
        t_inf = t2 - t1
        print(t_inf)
        print(len(trt_outputs))

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    print(len(trt_outputs))
    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(8,34),  (14,60),  (23,94),  (39,149),  (87,291),  (187,472)],
                          "obj_threshold": 0.1,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.3,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    print('test')
    for i in range(16):
        img_raw = raw[i]
        shape_orig_WH = img_raw.size
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH), i)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = draw_bboxes(img_raw, boxes, scores, classes, ALL_CATEGORIES)
        output_image_path = str(i+1) + '_416_bboxes.png'
        obj_detected_img.save(output_image_path, 'PNG')
        print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    main()
