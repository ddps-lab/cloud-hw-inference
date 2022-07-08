## Deep Learning Inference Serving
Object Detection on CPU 

### Environments
- Deep Learning AMI (Ubuntu 18.04) Version 60.1 (ami-0d6e58541939104ee)
- c6i.2xlarge
- python3.8
- tensorflow 2.8.0 


### Step 
```bash
#0. git clone 
git clone https://github.com/ddps-lab/cloud-hw-inference.git
cd cloud-hw-inference/CPU/object_detection

#1. make env 
source activate tensorflow2_p38
pip3 install tensorflow==2.8.0

#2. export model in tensorflow framework 
python3 export.py --data data/coco.yaml --include saved_model --batch-size 1

#3. inference model 
python3 val.py --data data/coco.yaml --weights yolov5s_saved_model/ --batch-size 1 --save-txt

```
