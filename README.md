# videoSlicer
Simple implementation of video editing in python using some CV and ML techniques to demonstrate what I can do with these tools.

videoSlicer uses OpenCV, ffmpeg to make what it does.

Special thanks to the authors of [keras-yolo3](https://github.com/qqwweee/keras-yolo3)

STEPS:
- Get video information
- Get audio from video with ffmpeg
- Sectorize image
![Alt text](images/splitting_in_sectors.png?raw=true "split")
- Apply basic transformations to sectors
![Alt text](images/transforming_sectors.png?raw=true "split")
- Apply YOLO (you only look once) object detection
![Alt text](images/yolo.png?raw=true "split")
- Get YOLO on video
![Alt text](images/yolo_video.png?raw=true "split")
- Join sectors
![Alt text](images/join.png?raw=true "split")
- Join audio with ffmpeg
