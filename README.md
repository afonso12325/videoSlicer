# videoSlicer
Simple implementation of video editing in opencv using some CV and ML techniques to demonstrate what I can do with these tools.

videoSlicer uses OpenCV, ffmpeg to make what it does.

STEPS:
- Get video information
- Get audio from video with ffmpeg
- Sectorize image
![Alt text](images/splitting_in_sectors.png?raw=true "split")
- Apply basic transformations to sectors
![Alt text](images/transforming_sectors.png?raw=true "split")
- Join sectors
![Alt text](images/join.png?raw=true "split")
- Join audio with ffmpeg