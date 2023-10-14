# Implementation YOLOv3 using PyTorch 
![image1](https://github.com/AlexeyDate/YOLOv3/assets/86290623/589006e4-88a2-4d3d-84b2-0d7b531cf060)

## YOLOv3 with CIoU loss 💡 
**In this version, convergence has been improved due to the implementation of CIoU loss for bounding box regression.
The main part of the code has been rewritten and optimized. Added the ability to select the image resolution for training** 

**Now you can observe the training process, having an informative graph for each loss.**

![loss_100](https://github.com/AlexeyDate/YOLOv3/assets/86290623/7b800efe-dd62-4152-85be-aa02bfabc389)

***This release v2.0 is recommended for all trainings*** ✅

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

## Clustering
* Before training, you should extract clusters of anchor boxes based on k-means clustering:
> 
    python3 clusterAnalysis.py --path ./data/obj.data --n_cluster 9 --gen 100

All clustering parameters:

`--path`                (states: path to obj.data)

`--n_cluster`           (states: clusters number)

`--gen`                 (states: generations number)

**Note**: Set the results obtained below to the `anchors` variable in `train.py` and `detect.py`
>	
    average IOU: 0.983
    anchor boxes:
	 [[ 76.65064664 107.10037741]
	 [168.95037498 172.72056937]
	 [134.74968954 247.5132047 ]
	 [219.54969625 240.28065663]
	 [167.14672785 327.9973526 ]
	 [261.83260957 280.90816227]
	 [333.50469046 293.58168127]
	 [285.5608964  381.0593046 ]
	 [384.12271692 380.82027335]]

## Training
* Moving on to training
> 
    python3 train.py --epochs 100 --lr 0.003
    
All training parameters:

`--image size`              (states: input image size)

`--epochs`                  (states: total epochs)

`--lr`           	    (states: learning rate)

`--batch_size`              (states: batch size)

`--weight_decay`            (states: weight decay)

`--weights`            	    (states: path to yolo PyTorch weights or path to Darknet53 binary weights if you want to train your model form scratch)

`--multiscale_off`          (states: disable multi-scale training)

`--verbose`		    (states: show all losses and resolution changes)

* After training, mAP will be calculated on the train dataloader and the test dataloader. 

**Note**: You can change the thresholds in `train.py`.

## Inference
On video:
> 
    python3 detect.py --image_size 416 --video --data_test content/video.mp4 --output content/detect.mp4 --weights backup/yolov3.pt
On image:
> 
    python3 detect.py --image_size 416 --data_test content/image.jpg --output content/detect.jpg --weights backup/yolov3.pt

Additional parameters:

`--show`          (states: show frames during inference)

**Note**: You can change the thresholds in `detect.py`.

![image2](https://github.com/AlexeyDate/YOLOv3/assets/86290623/f3f1021f-65ff-45cf-b5e0-2b58c4ffd03f)

## Comparison
| Model   		      		| Dataset 	     | Input size <br> <sub> (pixel)         | mAP <br> <sub>(@0.5) |
| :---:   		      		| :---:   	     | :---:    	                     | :---: 		    | 
| YOLOv1 <br> <sub> (Ours⭐)  		| African Wildlife   | 448       	                     | 61     	  	    |
| YOLOv2 <br> <sub> (Ours⭐)  		| African Wildlife   | 416       	      	             | 72    	            |
| YOLOv3 <br> <sub> (Ours⭐)  		| African Wildlife   | 416       	      	             | 77    	            |
| YOLOv3 CIoU loss <br> <sub> (Ours⭐)  | African Wildlife   | 416       	      	             | 82    	            |

## Test API ⚙
>
```python
from torch.utils.data import DataLoader
from dataloader.dataset import Dataset
from model.yolo import YOLOv3
from utils.utils import get_bound_boxes
from utils.mAP import mean_average_precision

anchors = [[[313, 303], [336, 323], [306, 371]],
           [[139, 222], [149, 212], [171, 205]],
           [[23, 34], [99, 234], [129, 224]]]

# Creating custom dataloaders for validation
# You can see an example of creation in 'train.py'
# ...

# Creating model and loading pretrained weights
model = YOLOv3(anchors=anchors, num_classes=classes).to(device)
model.load_state_dict(torch.load('path to weights'))  # Here you have to write the path to weights

# Getting prediction boxes and true boxes
# nms_threshold - threshold for non-maximum suppresion
# threshold - threshold in model prediction
pred_boxes, true_boxes = get_bound_boxes(dataloader, model, anchors, nms_threshold=0.5, threshold=0.3, device=device)

# As a result, we need to calculate mAP
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(mAP)
```

## Dependencies
**PyTorch** 
> Version: 1.13.1

**Albumentations**
> Version: 1.3.0

**OpenCV**
> Version: 4.7.0

**NumPy**
> Version: 1.23.0

**xmltodict**
> Version: 0.13.0

**terminaltables**
> Version: 3.1.10

## References
* [Original YOLOv3 paper](https://arxiv.org/pdf/1804.02767.pdf)
___
* [Darknet53 full weights from ImageNet](https://pjreddie.com/media/files/darknet53_448.weights) (recommended for all trainings)
* [Darknet53 convolutional weights from ImageNet](https://pjreddie.com/media/files/darknet53.conv.74)
___
* [African Wildlife dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife?resource=download)
* [African Wildlife PyTorch weights](https://drive.google.com/file/d/1-3e_b5LmlI2Ee9HEzg6e1UdcF1V2YK2M/view?usp=sharing)
* [African Wildlife optimizer state](https://drive.google.com/file/d/1-GKbCk-1fZCjbhe-lBIWqe1Fd8rWq_NT/view?usp=drive_link)

## Contact
* Developer: **Alexey Serzhantov**
* Email: serzhantov0289@gmail.com
