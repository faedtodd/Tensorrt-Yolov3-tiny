# Tensorrt-Yolov3-tiny

darknet.weights --> onnx.onnx --> tensorrt.trt

performance on Tx2 with turn fp16 mode on:
       
    input      batch size    inference time/ms
    416x416        1                9.5
                   16               8.75
                   32               8.5
    608x608        1                20
                   16               18.1
                   32               18
