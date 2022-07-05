## Deep Learning Inference Serving
Image Classification on CPU 

### Environments
- Deep Learning AMI (Ubuntu 18.04) Version 60.1 (ami-0d6e58541939104ee)
- c6i.2xlarge
- python3.8
- tensorflow 2.8.0 


### Step 
```bash
#0. git clone 
git clone https://github.com/ddps-lab/cloud-hw-inference.git
cd cloud-hw-inference/CPU/img_clf

#1. make env 
source activate tensorflow2_p38
pip3 install tensorflow==2.8.0
pip3 install awscli

#2. download ImageNet datasets
#2-1. aws configure 
mkdir datasets
cd datasets # path /home/ubuntu/datasets

aws s3 cp s3://--------- . 

#3. save pretrained model 
bash save_all.sh

#4. inference model 
# inference with 5 models and 1-64 batchsize   
bash inference_all.sh 

```
