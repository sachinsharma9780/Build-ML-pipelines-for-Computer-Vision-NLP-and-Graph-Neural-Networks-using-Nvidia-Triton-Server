# AI-Enterprise-Workshop-Building-ML-Pipelines
In this workshop we are going to use Nvidia's Triton Inference Server (formerly known as TensorRT Inference Server) 
which simplifies the deployment of AI models at scale in production. The one thing which attracted me the mdost is the capability of Triton inference server to host/deploy trained models from any framework (whether it is a TensorFlow, TensorRT, PyTorch, Caffe, ONNX, Runtime, or some custom framework) from local storage or Google Cloud Platform or AWS S3 on any GPU- or CPU-based infrastructure (cloud, data center, or edge). Therefore, for the purpose of this examination, we focus on hosting/deploying trained (on ImageNet) image classification models like InceptionNet, MobileNet etc on triton inference server. Once deployed we can make inference requests and can get back the predictions. 


![1_126iG2mnfl4i6iH9FKu3sg](https://user-images.githubusercontent.com/40523048/120965914-c4a98380-c765-11eb-86f0-eb2ce2574e97.png)
Image depicting the capability of [Nvidia's Triton Inference server](https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us) to host Multiple heterogeneous deep learning frameworks.

For, setting up the Triton inference server we generally need to pass two hurdles: 1) Set up our own inference server, and 2) After that, we have to write a python client-side script which can communicate with the inference server to send requests (in our case text) and get back predictions or image/text feature embeddings.

# Part1: Setting up Triton Inference Server on the machine
Let's start by setting up a triton server locally on the computer by following the below steps.

## Install Docker
[Docker](https://docs.docker.com/get-docker/)

## Pulling triton server docker image from Nvidia NGC:
1. [Download](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver/)
2. or use the command: docker pull nvcr.io/nvidia/tritonserver:21.05-py3
3. Image size: 10.6 GB (10-15 mins to install) 

## Create a model repository to add your models:
1. Clone the [Triton Inference Server GitHub repository](https://github.com/triton-inference-server/server.git
) if you need an example model repository (this will also download some pre-trained models structured in a manner as expected by Triton)
2. After cloning, you can find pre-trained models under: server → docs →examples →model_repository
3. Or you can clone this repo and in the model_repository folder I have already stored some default trained models whiich comes with cloning the above repsitory.
4. Instantiate triton server using the cmd: docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/example/model/repository:/models <docker image> tritonserver —model-repository=/models

  Note: Where <docker image> is nvcr.io/nvidia/tritonserver:<xx.yy>-py3 if you pulled the Triton container from NGC. -v flag points to the path of your model repository where all your models are stored as showed above.
