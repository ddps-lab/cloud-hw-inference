FROM amazonlinux:2

RUN echo $'[neuron] \n\
name=Neuron YUM Repository \n\
baseurl=https://yum.repos.neuron.amazonaws.com \n\
enabled=1' > /etc/yum.repos.d/neuron.repo

RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

RUN yum install -y git

RUN yum install -y aws-neuron-tools
RUN yum install -y aws-neuron-runtime
RUN yum install -y tar gzip

ENV PATH="/opt/aws/neuron/bin:${PATH}"
ENV NEURONCORE_GROUP_SIZES='4x1'

RUN yum install -y python3 python3-pip
RUN yum install -y python3.7-venv gcc-c++
RUN python3.7 -m venv tensorflow_venv
RUN source tensorflow_venv/bin/activate
RUN pip3 install -U pip

# Set Pip repository  to point to the Neuron repository
RUN pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

#Install Neuron TensorFlow
RUN pip install tensorflow-neuron[cc] "protobuf<4"
RUN pip install --upgrade "transformers==4.1.0"

RUN git clone https://github.com/jaeriver/cloud-hw-inference

