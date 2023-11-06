import tempfile
import cv2
import gradio as gr
from pathlib import Path
import os,shutil
import json
import gradio.components.gallery
from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
from argparse import ArgumentParser
from clusterer import RefClusterer
from logging import Logger
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
'''
overhaul:now only 1 stage,
'''
class gradio_ui(object):
    def __init__(self,args):
        self.args = args
        self.refextractor = RefClusterer(Path(args.model_dir))
        charas_path = Path("characters.json")
        if charas_path.exists():
            with open(charas_path,'r') as f:
                try:
                    self.refextractor.tagger.chara_tags = json.load(f)
                except:
                    pass
        self.last_folder = Path("output")
        self.last_mark_folder = Path("output").joinpath("mark_characters")
        self.chara_folders = {}
        self.logger = Logger("Logger")
    #postprocessing
    def make_grids(self,files:list,row,col,size,progress=gr.Progress()):
        output_folder = self.refextractor.get_task_dir("postprocess",["grids"])
        if not output_folder.exists():
            os.makedirs(output_folder)
        row = int(row)
        col = int(col)
        size = int(size)
        imgs = [ ]
        grids = [ ]
        for f in progress.tqdm(files):
            print(f.name)
            img = cv2.imread(f.name)
            img = self.refextractor.pad_image(img)
            img = cv2.resize(img,(size,size),cv2.INTER_CUBIC)
            imgs.append(img)

        group_size = row*col
        for i in range(0,len(imgs),group_size):
           chunk = imgs[i:min(i+group_size,len(imgs))]
           grid = self.refextractor.make_grid(chunk,row,col)
           cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),grid )
           grid = cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)
           grids.append(grid)
        return grids

    def line_option(self,selected:list):
        # "Gaussian","Laplacian","Neural Network"
        print("Selected:",selected)
        res = []
        if "Gaussian" in selected:
            res = res+[gr.Slider(visible=True,interactive=True),gr.Slider(visible=True,interactive=True),gr.Slider(visible=True,interactive=True)]
        else:
            res = res + [gr.Slider(visible=False, interactive=False),
                         gr.Slider(visible=False, interactive=False),
                         gr.Slider(visible=False, interactive=False)]
        if "Laplacian" in selected:
            res = res+[gr.Slider(visible=True,interactive=True)]
        else:
            res = res + [gr.Slider(visible=False, interactive=False)]
        if "Neural Network" in selected:
            res = res + [gr.Radio(visible=True, interactive=True)]
        else:
            res = res + [gr.Radio(visible=False, interactive=False)]
        print(len(res))
        return res
    def extract_lineart(self,files:list,selected:list[str],
                        dilate_it,dilate_ksize,gaussian_ksize,
                        laplacian_ksize,
                        nn_choice:str,
                        progress=gr.Progress()):
        '''
            line_folder_upload,
           line_process_options,
           line_gaussian_dilate_it, line_gaussian_dilate_ksize, line_gaussian_blur_ksize,
           line_laplacian_ksize,
           line_nn_choice
        '''
        output_folder = self.refextractor.get_task_dir("postprocess",["lineart"])
        if not output_folder.exists():
            os.makedirs(output_folder)

        if "Gaussian" in selected:
            self.refextractor.line_extractor.ksize_gaussian = gaussian_ksize
            self.refextractor.line_extractor.ksize_dilate = dilate_ksize
            self.refextractor.line_extractor.it_dilate = dilate_it
        if "Laplacian" in selected:
            self.refextractor.line_extractor.ksize_laplacian = laplacian_ksize
        lines = []
        for f in progress.tqdm(files):
            print(f.name)
            img = cv2.imread(f.name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if "Gaussian" in selected:
                print(img.shape)
                img = self.refextractor.line_extractor.gaussian(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            if "Laplacian" in selected:
                print(img.shape)
                img = self.refextractor.line_extractor.laplacian(img)
            if "Neural Network" in selected:# ["Anime2Sketch", "MangaLineExtraction"]
                if nn_choice=="Anime2Sketch":
                    print(img.shape)
                    img = self.refextractor.line_extractor.sketch_line(img)
                elif nn_choice=="MangaLineExtraction":
                    print(img.shape)
                    img = self.refextractor.line_extractor.manga_line_batch(img)
            # if img.
            img = cv2.normalize(img,img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),
                        img)
            lines.append(img)
            # line = self.refextractor.lineart(img,int(dilate_it),int(dilate_ksize),int(gaussian_ksize))
            # cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),
            #             line)
            # lines.append(line)
        return lines
    def upscale(self,files:list,upscale_scale:float,upscale_model:str,upscale_sharpen:bool,upscale_sharpen_mode:str,upscale_sharpen_ksize:float,progress=gr.Progress()):
        output_folder = self.refextractor.get_task_dir("postprocess",["upscale"])
        if not output_folder.exists():
            os.makedirs(output_folder)
        res = []
        for f in progress.tqdm(files):
            print(f.name)
            # img = cv2.imread(f.name)
            img_path = output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve()
            res_path = self.refextractor.upscaler.upscale(f.name,img_path,int(upscale_scale),upscale_model)
            # res_img = self.refextractor.upscaler.upscale_img(img,int(upscale_scale),upscale_model)
            if upscale_sharpen:
                res_img = cv2.imread(res_path.resolve().__str__())
                res_img = self.refextractor.upscaler.sharpen(res_img,upscale_sharpen_mode,upscale_sharpen_ksize)
                # res_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(res_path.resolve().__str__(),res_img)
            # res_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)
            # res.append(res_img)
            res.append(res_path.resolve().__str__())
            # this has to update iteratively since this is kinda slow
        return res

    def extract_similar_from_vid(self,video_path:Path,img_path:str,thresh=0.85):
        logging.info(msg=f"Start extracting similar images from video with threshold of {thresh}")
        video_path = Path(video_path)
        img_path = Path(img_path)
        target_img, sim_imgs = self.refextractor.extract_similar_from_vid(target_img_path=img_path,video_path=video_path,thresh=thresh)
        return sim_imgs
    def save_similar(self,target_img,sim_imgs):
        sim_imgs:gradio.components.gallery.GalleryData
        target_img = Path(target_img)
        target_img = Image.open(target_img)
        dir = self.refextractor.get_task_dir("match_chara")
        if sim_imgs is not None:
            img_paths = [Path(img.image.path) for img in sim_imgs.root]
        else:
            img_paths = []
        self.refextractor.save_similar(dir,target_img,img_paths)
        return f"Successfully saved {len(sim_imgs.root)} images to {dir}"

    def interface(self):
        # output_format = gr.Radio(choices=["imgs","video"],
        #                          value="imgs",
        #                          label="Output format",
        #                          info="If chosen 'imgs', the program will output a series of images. If chosen 'video', the program will output a video.",
        #                          interactive=True)
        # output_mode = gr.Radio(choices=["crop","draw","highlight"],
        #                        value="crop",
        #                        label="Output annotation mode",
        #                        info="If chosen 'crop', the program will output cropped images according to inference. If chosen 'draw' and 'highlight', the program will either draw or highlight the marked area on the original image.",
        #                        interactive=True)
        # model_selection = gr.Dropdown(choices=self.refextractor.models,
        #                               value=self.refextractor.models[0],
        #                               label="Model",
        #                               info="Which detection model to use. Models' sizes go from n->s->m->l. The larger the more accurate, but also slower.",
        #                               interactive=True)
        # threshold_slider = gr.Slider(minimum=0.0,maximum=1.0,value=0.2,step=0.05,label="Keyframe Threshold",info="Larger value means fewer keyframe extracted for imgs mode",interactive=True)
        # padding_slider = gr.Slider(minimum=-0.5,maximum=1.0,value=0.0,step=0.05,label="Detection Padding",info="Pad the detection boxes(optional)",interactive=True)
        # conf_threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5,step=0.05, label="Detection Confidence Threshold",
        #                              info="How confident the detection result has to be to be considered.", interactive=True)
        # iou_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05,
        #                                   label="Box Merging(IOU) Threshold",
        #                                   info="How large the intersection of boxes has to be to be merged.",
        #                                   interactive=True)
        # min_bbox_size_slider = gr.Number(label="Minimum Bounding Box Size",
        #                                  info="Detection result with shorter edge smaller than this size will not be considered.",
        #                                  value=300,precision=10)
        ref_img = gr.Image(label="Upload your reference image!",interactive=True,visible=True,type="filepath")
        sim_thresh_slider = gr.Slider(minimum=0.1,maximum=1.0,step=0.05,value=0.85,label="Similarity Threshold",
                                      info="How similar an image has to be to be considered similar to the target image",interactive=True)
        sim_save_btn = gr.Button(value="Save results",interactive=True,variant="primary")

        vid_upload = gr.Video(label="Upload your video!")
        vid_submit = gr.Button(value="Submit video!",variant="primary")

        res_imgs = gr.Gallery(label="Result",visible=True,columns=6,container=True)
        sim_msg = gr.Textbox(label="Message")

        grid_hint = gr.Textbox(label="Hint",value="Upload one/multiple image files ")
        grid_rows = gr.Slider(minimum=1,maximum=12,value=3,step=1,label="Grid Rows",interactive=True)
        grid_cols = gr.Slider(minimum=1, maximum=12, value=3, step=1, label="Grid Columns",interactive=True)
        grid_size = gr.Number(value=300,label="Grid size",info="Size of each image in the grid",interactive=True)
        grid_folder_upload = gr.File(label="Upload dataset",
                                     file_count="multiple",
                                     # info="Upload the folder with images you want to make into grids",
                                     interactive=True,elem_id="grid-files"
                                     )
        grid_submit_btn = gr.Button(value="Make grids",variant="primary",interactive=True)
        grid_res_gallery = gr.Gallery(label="Grid Result",columns=6)

        #extract line art
        line_folder_upload = gr.File(label="Upload dataset",
                                     file_count="multiple",
                                     # info="Upload the folder with images you want to extract lineart",
                                     interactive=True,elem_id="line-files"
                                     )
        line_process_options = gr.CheckboxGroup(choices=["Gaussian","Laplacian","Neural Network"],value="Gaussian",label="Line art extraction options",
                                                info="Choosing multiple options will result in img processed through each method sequentially.",interactive=True,visible=True)
        '''
        for gaussian method
        '''
        line_gaussian_dilate_it = gr.Slider(label="Iterations",
                                            info="More iteration usually gives better result, but slower.",
                                            minimum=1, maximum=20, value=1, step=1, interactive=True)
        line_gaussian_dilate_ksize = gr.Slider(label="Dilation kernel size",
                                               info="Higher value gives more lines but also more noises.",
                                               minimum=1, maximum=15, value=7, step=2, interactive=True)
        line_gaussian_blur_ksize = gr.Slider(label="Gaussian blur kernel size",
                                             info="You can play with this.",
                                             minimum=1, maximum=15, value=3, step=2, interactive=True)
        '''
        For Laplacian method 
        '''
        line_laplacian_ksize = gr.Slider(label="Laplacian kernel size",
                                            info="Larger value yields more lines, while more noise will be taken in",
                                            minimum=1, maximum=15, value=3, step=2, interactive=True,visible=False)
        '''
        For Neural Network method
        '''
        line_nn_choice = gr.Radio(label="Neural Network Model",choices=["Anime2Sketch", "MangaLineExtraction"],value="Anime2Sketch",interactive=True,visible=False)



        line_submit_btn = gr.Button(value="Extract lineart",variant="primary",interactive=True)
        line_res_gallery = gr.Gallery(label="Lineart Result",columns=6)

        #Upscaling
        upscale_scale = gr.Slider(label="Scale to",
                                  minimum=1,maximum=4,step=1,value=2)
        upscale_model = gr.Dropdown(label="Model",
                                    choices=self.refextractor.upscaler.models,
                                    value=self.refextractor.upscaler.models[0],
                                    interactive=True)
        upscale_folder_upload = gr.File(label="Upload dataset",
                                     file_count="multiple",
                                     # info="Upload the folder with images you want to upscale",
                                     interactive=True,elem_id="up-files"
                                     )
        upscale_sharpen = gr.Checkbox(label="Sharpen",
                                      value=False,
                                      info="Sharpen the images after upscaling",
                                      interactive=True)
        upscale_sharpen_mode = gr.Radio(label="Sharpen Mode",
                                        value="Laplace",
                                        choices=self.refextractor.upscaler.sharpen_modes,
                                        interactive=True)
        upscale_sharpen_ksize = gr.Number(label="Kernal Size",
                                          info="This only works for USM mode, higher means stronger effect.",
                                          value=1,
                                          interactive=True)
        upscale_submit_btn = gr.Button(value="Upscale images!", variant="primary", interactive=True)
        upscale_stop_btn = gr.Button(value="Stop",interactive=True)
        upscale_res_gallery = gr.Gallery(label="Upscale Result",columns=6)

        with gr.Blocks(title="AniRef",css="""
            .file-preview{
                max-height:20vh;
                overflow:scroll !important;
            }
        """) as demo:
            with gr.Tabs() as tabs:
                with gr.TabItem("Inference",id=0):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ref_img.render()
                            sim_thresh_slider.render()
                            sim_save_btn.render()
                        with gr.Column(scale=3):
                            vid_upload.render()
                            vid_submit.render()
                            sim_msg.render()
                            res_imgs.render()
                with gr.TabItem("Postprocessing",id=2):
                    with gr.Tabs(selected=0):
                        with gr.TabItem(label="Make grids",id=0):
                            with gr.Row():
                                grid_rows.render()
                                grid_cols.render()
                            with gr.Row():
                                grid_size.render()
                            with gr.Row():
                                grid_submit_btn.render()
                            with gr.Row():
                                grid_res_gallery.render()
                            with gr.Row():
                                grid_folder_upload.render()
                        with gr.TabItem(label="Extract lineart", id=1):
                            with gr.Row():
                                line_submit_btn.render()
                            with gr.Row():
                                line_process_options.render()
                            with gr.Row():
                                line_gaussian_dilate_it.render()
                                line_gaussian_dilate_ksize.render()
                                line_gaussian_blur_ksize.render()
                            with gr.Row():
                                line_laplacian_ksize.render()
                            with gr.Row():
                                line_nn_choice.render()
                            with gr.Row():
                                line_res_gallery.render()
                            with gr.Row():
                                line_folder_upload.render()
                        with gr.TabItem(label="Upscaling", id=2):
                            with gr.Row():
                                upscale_submit_btn.render()
                                # upscale_stop_btn.render()
                            with gr.Row():
                                with gr.Row():
                                    upscale_scale.render()
                                    upscale_model.render()
                                with gr.Row():
                                    upscale_sharpen.render()
                                    upscale_sharpen_mode.render()
                                    upscale_sharpen_ksize.render()
                            with gr.Row():
                                upscale_res_gallery.render()
                            with gr.Row():
                                upscale_folder_upload.render()

            '''Event handlers'''
            vid_submit.click(fn=self.extract_similar_from_vid,inputs=[vid_upload,ref_img,sim_thresh_slider],outputs=[res_imgs])
            sim_save_btn.click(fn=self.save_similar,inputs=[ref_img,res_imgs],outputs=[sim_msg])

            grid_submit_btn.click(fn=self.make_grids,inputs=[grid_folder_upload,grid_rows,grid_cols,grid_size],outputs=[grid_res_gallery])
            line_submit_btn.click(fn=self.extract_lineart, inputs=[line_folder_upload,
                                                                   line_process_options,
                                                                   line_gaussian_dilate_it, line_gaussian_dilate_ksize, line_gaussian_blur_ksize,
                                                                   line_laplacian_ksize,
                                                                   line_nn_choice
                                                                   ], outputs=[line_res_gallery])
            upscale_event =  upscale_submit_btn.click(fn=self.upscale,inputs=[upscale_folder_upload,upscale_scale,upscale_model,upscale_sharpen,upscale_sharpen_mode,upscale_sharpen_ksize],outputs=[upscale_res_gallery])

            line_process_options.change(fn=self.line_option,inputs=[line_process_options],outputs=[line_gaussian_dilate_it,line_gaussian_dilate_ksize,line_gaussian_blur_ksize,
                                                                                                   line_laplacian_ksize,
                                                                                                   line_nn_choice])
        if args.port is None:
            demo.launch(share=args.share,inbrowser=args.inbrowser)
        else:
            demo.launch(share=args.share, inbrowser=args.inbrowser,server_port=args.port)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--share",action="store_true",default=False,help="Expose the web app to public.")
    parser.add_argument("--inbrowser",action="store_true",default=False,help="Open the web app in browser.")
    parser.add_argument("--port",type=int,nargs="?",help="Specify the server port, if not specified, then the app will assign a random port.")
    parser.add_argument("--model_dir",default="models",type=str,help="Specify the models folder containing yolov8 and line art models.")
    args = parser.parse_args()
    ui = gradio_ui(args)
    ui.interface()