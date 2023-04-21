# Implementation YOLOv3 using PyTorch 

## Dataset
* This repository was train on the [African Wildlife Dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) from Kaggle
* The data folder must contain the train and test folders as follows:
> 
    ├── data 
      ├── train
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
      ├── test 
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
      ├── obj.data
      ├── obj.names

* Also, for training, in the data folder there must be `obj.data` file with some settings
>
    classes = 4
    train = data/train
    valid = data/test
    names = data/obj.names
    backup = backup/
    file_format = txt
    convert_to_yolo = False
    
* And there must be `obj.names` file with label names.
>
    0 buffalo
    1 elephant
    2 rhino
    3 zebra

* In the description files, you can write the coordinates of the bounding boxes in a simple format `(x1, y1, x2, y2)`. After this use the appropriate flag when training. YOLO format is also available and recommended. Format files as follows:    
    
**txt**
>
    <class> <xmin> <ymin> <xmax> <ymax>
example:
>
    1 207 214 367 487
___
**txt** (already converted to yolo)
>
    <class> <xcenter> <ycenter> <width> <height>
example:
>
    1 0.2 0.3 0.15 0.23
___
**xml**

example:
>
    <annotation>
	<object>
		<name>zebra</name>
		<bndbox>
			<xmin>71</xmin>
			<ymin>60</ymin>
			<xmax>175</xmax>
			<ymax>164</ymax>
		</bndbox>
    
## Training
* Before training, you should extract clusters of anchor boxes based on k-means clustering:
> 
    python3 clusterAnalysis.py
**Note**: Set the results obtained below to the `anchors` variable in `train.py` and `detect.py`
>	
    average IOU: 0.983
    anchor boxes:
     [[0.11225179 0.20962439]
     [0.2432669  0.25324393]
     [0.33063245 0.37442702]
     [0.3273001  0.51367768]
     [0.26461367 0.7279446 ]
     [0.61434309 0.51579031]
     [0.44362939 0.72581586]
     [0.77726385 0.7311893 ]
     [0.71958308 0.90361056]]



* Moving on to training
> 
    python3 train.py --epochs 100 --learning_rate 1e-4 
    
All training parameters:

`--epochs`                  (states: total epochs)

`--learning_rate`           (states: learning rate)

`--batch_size`              (states: batch size)

`--weight_decay`            (states: weight decay)

`--yolo_weights`            (states: path to yolo PyTorch weights)

`--darknet_weights`         (states: path to extraction binary weights, it's base CNN module of YOLO)

`--multiscale_off`          (states: disable multi-scale training)

`--verbose`		    (states: show all losses and resolution changes)

* After training, mAP will be calculated on the train dataloader and the test dataloader. 

**Note**: You can change the thresholds in `train.py`.

## Inference
On video:
> 
    python3 detect.py --video --data_test content/video.mp4 --output content/detect.mp4 --weights backup/yolov1.pt
On image:
> 
    python3 detect.py --data_test content/image.jpg --output content/detect.jpg --weights backup/yolov1.pt

Additional parameters:

`--show`          (states: show frames during inference)

**Note**: You can change the thresholds in `detect.py`.

## Comparison
| Model   		      | Dataset 	   | Input size <br> <sub> (pixel)    | mAP <br> <sub>(@0.5)   |
| :---:   		      | :---:   	   | :---:    	                      | :---: 		       | 
| YOLOv1 <br> <sub> (Ours⭐)  | African Wildlife   | 448       	                     | 61     	  	      |
| YOLOv2 <br> <sub> (Ours⭐)  | African Wildlife   | 416       	      	             | 72    	            |
| YOLOv3 <br> <sub> (Ours⭐)  | African Wildlife   | 416       	      	             | 63    	            |

***Why YOLOv3 has a worse metric than YOLOv2?***

*Now I can say that I have difficulty training YOLOv3 on this dataset because of strong overfitting. In the near future, I will conduct a lot of experiments and aim to get a decent result.*

## Dependencies
**PyTorch** 
> Version: 1.13.1

**Albumentations**
> Version: 1.3.0

**OpenCV**
> Version: 4.7.0

**xmltodict**
> Version: 0.13.0

## References
* [Original YOLOv3 paper](https://arxiv.org/pdf/1804.02767.pdf)
___
* [Darknet53 weights from ImageNet](https://pjreddie.com/media/files/darknet53_448.weights) (recommended for all trainings)
___
* [African Wildlife dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife?resource=download)
* [African Wildlife PyTorch weights](https://drive.google.com/file/d/1--mYXxFWhSqkNiIy3WtnzkrZRo4pIsnr/view?usp=sharing)
* [African Wildlife optimizer state](https://drive.google.com/file/d/1-0VeU6WTTxKwRfAYY8U3kAJ6XLIk5IIP/view?usp=sharing)
