# emhm

## Prerequisites

 - Python
 - (Python modules)
    - mediapipe (`pip install mediapipe`)
    - opencv (Automatically installed with mediapipe)


## Platforms

 - (Tested) Windows 10 64bit, Python 3.10 (anaconda), MediaPipe 0.9.0
 - (Should work) Linux, macOS, etc


## How to use

To process the video:
```
python emhm.py -i SOURCE_VIDEO.mp4 -o DEST_VIDEO.mp4
```

To print help:
```
python emhm.py --help
```


## Limitations

 - All operation in MediaPipe is executed in CPU. (GPU accelaration is disable in MediaPipe for Python)
 - MediaPipe can detect only one person. If multiple person exists in a frame, the result may be unstable.


# License

MIT License.
