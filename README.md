# AniRef2-yolov8
![demo_img](https://github.com/SoulflareRC/Aniref2-yolov8/assets/107384280/f705bab4-5fed-4a80-9cce-57a2172cde70)
**This project is an improved version of [Aniref-yolov8](https://github.com/SoulflareRC/AniRef-yolov8)**
### What does AniRef do?
**Given a reference image of a character and a video, it automatically extracts images of the character from the video.** <br> <br>
Finding good reference images has always been a challenging and time consuming task for artists. Most of the times it's tedious and hard to manually get reference images from the anime outselves and we end up relying on the artbooks published by the anime studios. This project mainly presents a toolchain for artists to quickly extract reference images of their desired characters from anime videos.  We first use an object detection model to crop out the characters, and then use [CLIP](https://github.com/openai/CLIP) to extract features of target image and extracted character crops, and finally find images of the target character by computing cosine similarity between target vector and character crop feature vectors.  
##### Currently Supported Features
- Character Detection
- Character Identification
- Image Restoration using [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- Image Grids
- Line Art Extraction
### Model
We trained an object detection model for detecting anime characters based on the state-of-the-art [YOLOv8](https://github.com/ultralytics/ultralytics/tree/main). We provide 4 models that are based on 4 sizes of YOLOv8 and all of them are trained on hand-annotated dataset focused on anime screenshots. Detailed metrics of each model can be found in [validation_metrics](https://github.com/SoulflareRC/AniRef-yolov8/tree/main/validation_metrics) <br>
| Model                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>test<br>50-95 | Speed<br><sup>RTX 3060 12G<br>(ms)  | 
| ------------------------- | --------------------- | -------------------- | ----------------------| ----------------------------------- | 
| [AniRef40000-n-epoch40](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-n-epoch40.pt) | 640                   | 49.9                 | 45.9                  | 3.0                                 | 
| [AniRef40000-s-epoch40](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-s-epoch40.pt) | 640                   | 52.2                 | 50.6                  | 6.4                                 | 
| [AniRef40000-m-epoch75](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-m-epoch75.pt) | 640                   | 50.2                 | 48.2                  | 15.0                                |
| [AniRef40000-l-epoch50](https://github.com/SoulflareRC/AniRef-yolov8/releases/download/model/AniRef40000-l-epoch50.pt) | 640                   | 52.9                 | 50.5                  | 23.1                                | 
### Known Issues
  - Performance of the models worsens when there exists too many overlapping characters. 
  - AniRef40000-m-epoch75 was trained on an older version of the dataset for 40 epochs and needs a retrain. 
### Dataset
The lastest raw dataset is now uploaded on [Kaggle](https://www.kaggle.com/datasets/ruochongchen69/anidet-7000). The dataset is first collected with [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter) on keyframes of anime compilation videos from the internet, and then manually corrected on Roboflow. The most recent version includes 10k images(40k after augmentation) and the datasets are available on [google drive](https://drive.google.com/drive/folders/1q1F1pJhRNboJkdi8XVVRiL7-_aeBFvTh?usp=share_link).
### Installation
1. Clone this repository 
``` 
git clone git@github.com:SoulflareRC/Aniref2-yolov8.git
cd Aniref2-yolov8
```
2. Create a virtual environment(optional)
``` 
python -m venv venv 
venv\Scripts\activate
```
3. Install the requirements
```
pip install -r requirements.txt
```
4. Run the UI and enjoy!
```
python gradio_interface.py
```
### Usage
See the [project wiki](https://github.com/SoulflareRC/AniRef-yolov8/wiki) for detailed instructions on how to use each function of the app. We also provide a demo video for how to use this program. 


