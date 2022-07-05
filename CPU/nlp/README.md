## Deep Learning Inference Serving
NLP on CPU 

### Environments
- Deep Learning AMI (Ubuntu 18.04) Version 60.1 (ami-0d6e58541939104ee)
- c6i.2xlarge
- python3.8
- tensorflow 2.8.0 


### Step 
```bash
#0. git clone 
git clone https://github.com/ddps-lab/cloud-hw-inference.git
cd cloud-hw-inference/CPU/nlp

#1. make env 
source activate tensorflow2_p38
pip3 install tensorflow==2.8.0
pip3 install transformers

#2. if save pretrained model BERT  
# with random dataset : seq_length 128 , batchsize 1-64
python3 bert_inference.py --save True 

#2-1. only inference BERT  
python3 bert_inference.py 

```
