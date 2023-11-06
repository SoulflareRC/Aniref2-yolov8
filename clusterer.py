import os
import time
from datetime import datetime
import shutil
import torch.cuda
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CFG
import hashlib
import json
import torch.nn as nn
import gdown
from utils.lineart_extractor import *
from utils.upscaler import Upscaler
from utils.extract_frames import *
from utils.tagger import Tagger
from yolov8.dataset import *
from transformers import  CLIPModel, CLIPProcessor
import torchvision
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RefClusterer(object):
    def __init__(self,model_path = None,models_folder = Path("models")):
        if model_path:
            self.model_path = model_path
        self.models_folder = models_folder
        self.models = self.grab_models(self.models_folder)
        self.model_path =self.models_folder.joinpath("yolov8").joinpath(self.models[0]+".pt" if ".pt" not in self.models[0] else self.models[0])
        self.model:YOLO = YOLO(self.model_path).to(device)
        self.extractor = Extractor(video=None,output_dir="temp")

        self.output_dir:Path = Path("output")
        self.save_img_format = ".jpg"
        # clip
        self.clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # for upscaling
        self.upscaler = Upscaler()

        # for line art
        self.line_extractor = LineExtractor()
        self.line_extractor.manga_model_path = "models/lineart/manga.pth"
        self.line_extractor.sketch_model_path = "models/lineart/sketch.pth"

        # caching similar extraction results
        self.last_vid_path = None
        self.last_img_features = None
        self.last_vid_dir = None
        self.last_img_paths = None
    def get_task_dir(self,task_name:str,sub_tasks:list[str]=[])->Path:
        if hasattr(self,"output_dir") or not self.output_dir:
            self.output_dir = Path("output")
        task_dir = self.output_dir
        task_dir = task_dir.joinpath(task_name)
        for task in sub_tasks:
            task_dir = task_dir.joinpath(task)
        task_dir = task_dir.joinpath(datetime.now().__str__().replace(":", ""))
        if not task_dir.exists():
            task_dir.mkdir(parents=True)
        return task_dir
    def grab_models(self,models_folder:Path):
        # models_folder = Path("models")
        if not models_folder.exists():
            models_folder_link = "https://drive.google.com/drive/folders/19Cnkg0y7kYq2uyC05E1DdLX24EEZjGlK?usp=share_link"
            gdown.download_folder(url=models_folder_link)
        models_folder = models_folder.joinpath("yolov8")
        models = [ x.stem for x in list(models_folder.iterdir())]
        print(f"Grabbed {len(models)} models:")
        for model in models:
            print(model)
        return models
    def extract_chara(self,video_path,frame_diff_threshold = 0.2,padding=0.0,conf_threshold=0.5,iou_threshold=0.6,min_bbox_size=400)->Path:
        '''
        :param video_path: path to video, extract keyframes and detect result
        :param mode:frame diff threshold is used for extracting keyframes
        :return:
        a list of images
        '''
        logging.info(msg=f"Extracting character crops with {self.model_path}")
        video = Path(video_path)
        if video.exists():
            res_paths = []
            if self.model == None:
                self.model = YOLO(self.model_path)
            # target_folder = self.output_dir.joinpath("video_to_imgs").joinpath(mode).joinpath(video.stem) # don't use video.stem since cv2 fails to write when path has special char
            target_folder = self.get_task_dir("extract_chara")
            if not target_folder.exists():
                os.makedirs(target_folder)
            self.extractor.video = video_path
            # keyframes:list[Frame] = self.extractor.extract_keyframes(frame_diff_threshold) # this will blow up the memory
            frame_fnames:list[Path] = self.extractor.extract_keyframes2(frame_diff_threshold)
            res_list =[]
            for frame_fname in tqdm(frame_fnames):
                frame_fname:Path
                # print(frame_fname)
                frame = cv2.imread(frame_fname.resolve().__str__())
                frame_fname.unlink(missing_ok=True)
                res:list[Results] = self.model.predict(frame,iou=iou_threshold,conf=conf_threshold)
                boxes = get_boxes(res,min_bbox_size=min_bbox_size)
                boxes = pad_boxes(frame,boxes,scale=padding)
                res_imgs = crop_boxes(frame,boxes)
                for res_img in res_imgs:
                    target_path = target_folder.joinpath(f"{len(list(target_folder.iterdir()))}{self.save_img_format}").__str__()
                    print(res_img.shape)
                    print(target_path)
                    cv2.imwrite(target_path,res_img)
            torch.cuda.empty_cache()
            return target_folder
        else:
            print("Video ",video_path," doesn't exist")
    def _get_image_features(self,image_paths):
        '''get image features for a batch'''
        # Load all the images in a batch
        images = [Image.open(image_path) for image_path in image_paths]

        # Use the processor to encode the batch of images
        inputs = self.clip_processor(text=None, images=images, return_tensors="pt", padding=True)
        pixel_val = inputs['pixel_values'].to(device)
        print("Processed shape:",pixel_val.shape)
        # Get image features from the model
        image_features = self.clip_model.get_image_features(pixel_values=pixel_val)
        image_features = image_features.detach().cpu()
        del images
        del inputs
        del pixel_val
        return image_features
    def get_image_features(self,img_paths:list,batch_size=1):
        '''get all image features from a list of path of images'''
        # torch.cuda.empty_cache()
        image_batches = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)]
        img_features = []
        for batch in image_batches:
            batch_features = self._get_image_features(batch)
            img_features.append(batch_features)
        all_features = torch.cat(img_features, dim=0)
        return all_features
    def cluster_chara(self,input_folder:Path,batch_size=1,thresh=5,min_img=2):
        torch.cuda.empty_cache()
        img_dir = Path(input_folder)
        img_paths = [f for f in img_dir.iterdir()]
        all_features = self.get_image_features(img_paths,batch_size)
        clusters = hcluster.fclusterdata(all_features, thresh, criterion="distance")
        max_class = clusters.max()
        each_class = []
        for i in range(1, max_class + 1):
            idx = np.where(clusters == i)[0]
            print(idx)
            if (len(idx) >= min_img):
                imgs = [img_paths[k] for k in idx]
                each_class.append(imgs)
        print(each_class)
        self.save_clusters(each_class,Path("cluster_result"))
        # fig, ax = plt.subplots(len(each_class), 8, figsize=(8, 8))
        # for i in range(len(each_class)):
        #     class_imgs = each_class[i]
        #     for j in range(8):
        #         if j < len(class_imgs):
        #             ax[i][j].imshow(Image.open(class_imgs[j]))
        #         ax[i][j].axis("off")
        # plt.savefig("cluster_result.png")
    def save_clusters(self,img_clusters:list,target_dir:Path):
        target_dir = Path(target_dir)
        if not target_dir.exists():
            target_dir.mkdir()
        for idx,cluster in enumerate(img_clusters):
            cluster_dir = target_dir / Path(str(idx))
            if not cluster_dir.exists():
                cluster_dir.mkdir()
            for img_idx,img in enumerate(cluster):
                target_img_path = cluster_dir / Path(str(img_idx)+".jpg")
                source_img_path = Path(img)
                shutil.copy(source_img_path,target_img_path)
        return target_dir
    def find_similar(self,idx,all_features:torch.Tensor,thresh=0.9)->torch.Tensor:
        '''idx:target image index all_features:all image features
            return: a Tensor containing all the images with similarity to target over threshold'''
        cos_sim = nn.CosineSimilarity()
        target = all_features[idx].unsqueeze(0)
        sim = cos_sim(target,all_features)
        res_ids = torch.argwhere(sim>thresh).squeeze()
        return res_ids
    def extract_similar(self,target_img_path:Path,img_paths:list,thresh=0.85):
        '''
        extracts the images similar to target_img_path from img_paths
        target_img: reference
        imgs: a list of PIL Images
        return: target image(PIL Image), similar images(list of PIL image)
        '''
        target_feature = self.get_image_features([target_img_path])
        if img_paths == self.last_img_paths:
            logging.info(msg="Reusing features from last run")
            img_features = self.last_img_features
        else:
            img_features = self.get_image_features(img_paths,batch_size=4)
        self.last_img_paths = img_paths
        self.last_img_features = img_features
        all_features = torch.cat([target_feature,img_features],dim=0)
        ids = self.find_similar(0,all_features,thresh)
        all_paths = [target_img_path,*img_paths]
        sim_paths = [all_paths[idx] for idx in ids]
        sim_imgs = [Image.open(f) for f in sim_paths]
        target_img = Image.open(target_img_path)
        return target_img,sim_imgs
    def extract_similar_from_vid(self,target_img_path:Path,video_path:Path,thresh=0.85,**kwargs):
        if video_path == self.last_vid_path:
            logging.info(msg="Reusing extraction result from last video")
            res_dir = self.last_vid_dir
        else:
            res_dir = self.extract_chara(video_path=video_path,**kwargs)
        self.last_vid_path = video_path
        self.last_vid_dir = res_dir # caching last video dir and path
        img_paths = [f for f in res_dir.iterdir()]
        target_img, sim_imgs = self.extract_similar(target_img_path,img_paths,thresh=thresh)
        return target_img, sim_imgs
    def save_similar(self,target_dir:Path,target_img:Image,sim_imgs:list):
        if len(sim_imgs) < 1:
            logging.warning(msg="No image to be saved")
        if not target_dir.exists():
            target_dir.mkdir()
        sim_dir = target_dir / "similar_images"
        target_img_path = target_dir / "target.jpg"
        if not sim_dir.exists():
            sim_dir.mkdir()
        # print(f"Target img:{target_img}")
        if type(target_img) == Path:
            target_img:Path
            target_img_path = target_dir / "target"+str(target_img.suffix)
            shutil.copy(target_img,target_img_path)
        else:
            # print("Target img is a pil image")
            target_img.save(target_img_path)

        if  type(sim_imgs[0] == Path):
            for idx, img in enumerate(sim_imgs):
                sim_path = sim_dir.joinpath(str(idx) + ".jpg")
                shutil.copy(img,sim_path)
        else:
            for idx, img in enumerate(sim_imgs):
                sim_path = sim_dir.joinpath(str(idx) + ".jpg")
                img.save(sim_path)

        return target_dir

    '''Postprocessing '''
    def lineart(self,img:np.ndarray,it_dilate=1, ksize_dilate=7,ksize_gausian=3)->np.ndarray:

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        blurred_img = cv2.GaussianBlur(img_gray, (ksize_gausian, ksize_gausian), 0)  # remove noise from image
        kernel = np.ones((ksize_dilate, ksize_dilate), np.uint8)
        img_dilated = cv2.dilate(blurred_img, kernel, iterations=it_dilate)  # raising the iterations help with darkness
        img_diff = cv2.absdiff(img_dilated, img_gray)
        # img_diff = cv2.absdiff(blurred_img, img_gray)
        contour = 255-img_diff
        # contour = cv2.erode(contour,(1,1),iterations=1)
        # contour = np.clip(contour*1.5,0,255)
        # ret,thresh = cv2.threshold(contour,240,255,cv2.THRESH_BINARY)
        dark = np.where(contour<240)
        contour[dark] = contour[dark] * 0.7
        return contour
    def pad_image(self,img:np.ndarray,padding=0):
        # take in cv2 image, pad to square
        h, w, c = img.shape
        if w > h:
            border_h = int((w - h) / 2)
            res = cv2.copyMakeBorder(src=img, left=padding, right=padding, top=border_h+padding, bottom=border_h+padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        elif h>w:
            border_w = int((h - w) / 2)
            res = cv2.copyMakeBorder(src=img, left=border_w+padding, right=border_w+padding, top=padding, bottom=padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        else:
            res = cv2.copyMakeBorder(src=img, left= padding, right=padding, top=padding,
                                     bottom=padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        return res
    def make_grid(self,imgs:list[np.ndarray], rows=3,cols=3):
        group_size = rows*cols
        for i in range(group_size-len(imgs)):
            white = np.full_like(imgs[0],255)
            imgs.append(white)
        for i in range(rows):
            if i == 0 :
                res =np.hstack(imgs[i*cols:(i+1)*cols])
            else:
                row = np.hstack(imgs[i*cols:(i+1)*cols])
                res = np.vstack((res,row))

        # res = res.reshape((900,900,3))
        print(res.shape)
        # res = res.reshape((rows,cols))
        return res

'''
extracting similar images involes several steps: 
1. extract keyframes (relatively fast) 
2. extract character crops (pretty slow) 
3. extract image features (really slow) 
4. cosine similarity get sim images (fast due to batching) 

what can be cached: 
1. target image (a bit meaningless, since it's fast to process just one image) 
2. keyframes (a bit meaningless as well, since it takes lots of space and it's fast to extract) 
3. chara crops (yes, doesn't take that much space but slow to extract) 
4. features (yes, takes a lot of time to extract) 

solution: reuse features from last time if input video is not changed, completely cut the whole extract chara crops thing,   


'''


