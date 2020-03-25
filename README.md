# Instrument-Playing Action Detection in Videos


### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```



### Run the code & get the results & retrive the final model (in google drive)
You can directly download the pretrained model and the preprocessed music instrument UCF dataset here [https://drive.google.com/drive/folders/1hg_n3NCw2j2msYdzEmdRHBunCgM4J3qX?usp=sharing](https://drive.google.com/drive/folders/1hg_n3NCw2j2msYdzEmdRHBunCgM4J3qX?usp=sharing). You need to unzip the folder and put all of the content in the unfolded directory including “video” and “pretrain” into your root folder and change the root path in the code in order to run your code.
To run the code, you can follow the youtube video [https://youtu.be/59WkfOapAj8](https://youtu.be/59WkfOapAj8) to run the jupyter notebook “CSCE689 HW4 Song 625007598 Training Script.ipynb” and get your re- sults (train.log, valid.log) in the “results1” folder (you may need to make a new directory named “results1”) or you can also run the “mymain.py” file in the command line. Also, you can directly get all the log result from the Google drive folder “results1” [https://drive.google.com/drive/folders/1hg_n3NCw2j2msYdzEmdRHBunCgM4J3qX?usp=sharing](https://drive.google.com/drive/folders/1hg_n3NCw2j2msYdzEmdRHBunCgM4J3qX?usp=sharing). The final model is also there with name: "save_50.pth", meaning it is the model trained with 50 epochs. The required figures could be get by running the jupyter script “Generate Figure.ipynb” or you can also check it in the “figs” and “figs2” folders.

### Note
The codes are only tested on ubuntu 16.04 system with 2 GPU (GeForce RTX 2080 Ti), CUDA Version: 10.2, torch 1.4.0. You should specify the GPU environment in the code line ``os.environ['CUDA_VISIBLE_DEVICES']='0,1'`` in order to make it work. If you you one GPU, for example, ``os.environ['CUDA_VISIBLE_DEVICES']='0', you should use smaller batch size in order to make the memory available. 