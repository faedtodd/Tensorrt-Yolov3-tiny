# Tensorrt-Yolov3-tiny

    darknet.weights --> onnx.onnx --> tensorrt.trt

    sample from TensorRT-5.x.x.x/samples/python/yolov3_onnx/

device: nvidia tx2

environment:  ubuntu18.04  
              tensorrt5.0.6.3  
              cuda10.0  
              cudnn7.3.1  


set input size and batch size

performance on Tx2 with turn fp16 mode on:
       
    input      batch size    inference time/ms
    416x416        1                9.5
                   16               8.75
                   32               8.5
    608x608        1                20
                   16               18.1
                   32               18
