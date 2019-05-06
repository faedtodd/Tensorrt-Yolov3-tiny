# Tensorrt-Yolov3-tiny

    darknet.weights --> onnx.onnx --> tensorrt.trt

    sample from TensorRT-5.x.x.x/samples/python/yolov3_onnx/

## Environment
device:  
nvidia tx2

environment:  
ubuntu18.04  
tensorrt5.0.6.3  
cuda10.0  
cudnn7.3.1  

## How to use
### set input size and batch size
    set labels: line14,line19|data_processing.py
    set output_layers: line723|yolov3_to_onnx.py
    set batch_size: batch|xxx.cfg, line62|yolov3_to_onnx.py, line140|onnx_to_tensorrt.py
    set input_size: input|xxx.cfg, line63|yolov3_to_onnx.py, line139|onnx_to_tensorrt.py
    set mode: line141|onnx_to_tensorrt.py

## performance on Tx2 with turn fp16 mode on:
       
    input      batch size    inference time/ms
    416x416        1                9.5
                   16               8.75
                   32               8.5
    608x608        1                20
                   16               18.1
                   32               18
