# AI-Enterprise-Workshop-Building-ML-Pipelines
In this workshop we are going to use [Nvidia's Triton Inference server](https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us) (formerly known as TensorRT Inference Server) 
which simplifies the deployment of AI models at scale in production. It natively supports multiple framework backends like TensorFlow, PyTorch, ONNX Runtime, Python, and even custom backends. It supports different types of inference queries through advanced batching and scheduling algorithms, supports live model updates, and runs models on both CPUs and GPUs. Triton is also designed to increase inference performance by maximizing hardware utilization through concurrent model execution and dynamic batching. Concurrent execution allows you to run multiple copies of a model, and multiple different models, in parallel on the same GPU. Through dynamic batching, Triton can dynamically group together inference requests on the server-side to maximize performance. Therefore, for the purpose of this examination, we focus on hosting/deploying multiple trained models (Tensorflow, PyTorch) on triton inference server in order to leverage its full potential. Once models are deployed we can make inference requests and can get back the predictions. 

# Agenda
<ul> 
<li> Introduction to <a href="https://www.arangodb.com/"> ArangoDB </a> and Nvidia's Triton Inference Server (Need, features, applications, etc.) </li> 
<li> Setting up Triton Inference server on local machine </li> 
<li> Deploy your first trained model (Tensorflow) with an application to image classification on triton inference server </li> 
<li> Deploy almost any Hugging Face PyTorch models with an application to zero-short text classification on triton inference server (Here we will convert given PyTorch models to triton acceptable models) </li> 
<li> Once models are deployed, we can write a python-client side script to intertact with triton server (i.e. sending requests and reciving back the predictions </li> 
<li>  Exploring the python image_client.py script in order to make image classification request </li> 
<li> Writing down your own client-side script to interact with Hugging Face NLP Models </li> 
<li>  <a href="https://github.com/triton-inference-server/server/blob/main/docs/metrics.md"> Triton Metrics</a> </li> 
<li> Storing inference results in ArangoDB using <a href="https://github.com/joowani/python-arango"> python-arango</a> 
</ul>



![1_126iG2mnfl4i6iH9FKu3sg](https://user-images.githubusercontent.com/40523048/120965914-c4a98380-c765-11eb-86f0-eb2ce2574e97.png)
Image depicting the capability of [Nvidia's Triton Inference server](https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us) to host Multiple heterogeneous deep learning frameworks.

For, setting up the Triton inference server we generally need to pass two hurdles: 1) Set up our own inference server, and 2) After that, we have to write a python client-side script which can communicate with the inference server to send requests (in our case text) and get back predictions or image/text feature embeddings.

# Part1: Setting up Triton Inference Server on the machine
Let's start by setting up a triton server locally on the computer by following the below steps.

### Quickstart with Docker
```
1. Install Docker
2. docker pull nvcr.io/nvidia/tritonserver:21.05-py3
3. git clone https://github.com/sachinsharma9780/AI-Enterprise-Workshop-Building-ML-Pipelines.git
4. cd ./AI-Enterprise-Workshop-Building-ML-Pipelines
5. docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models
6. curl -v http://localhost:8000/v2/health/ready

Continue to Part 2 below..
```

## Install Docker
[Docker](https://docs.docker.com/get-docker/)

## Pulling triton server docker image from Nvidia NGC:
1. [Download](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver/) docker image
2. or use the command: 
``` docker pull nvcr.io/nvidia/tritonserver:21.05-py3 ```
4. Image size: 10.6 GB (10-15 mins to install) 
5. To view the downloaded docker image: 
``` docker images ```

## Create a model repository to add your models:
1. Clone the [Triton Inference Server GitHub repository](https://github.com/triton-inference-server/server.git
) if you need an example model repository (this will also download some pre-trained models structured in a manner as expected by Triton)
2. After cloning, you can find the trained models under: server → docs →examples →model_repository
3. Or you can clone this repo and in the model_repository folder, I have already stored some default trained models with their corresponding configuration file which comes along while cloning the above repository.
4. Instantiate triton server using the cmd: </br>
``` docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/example/model/repository:/models docker image tritonserver —model-repository=/models ```

Note: Where docker image is nvcr.io/nvidia/tritonserver:<xx.yy>-py3 if you pulled the Triton container from NGC. -v flag points to the path of your model repository where all your models are stored as showed above.

e.g. ``` docker run  --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Users/sachin/Desktop/arangodb/scripts/triton/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models ```


![Screenshot 2021-06-07 at 11 24 57](https://user-images.githubusercontent.com/40523048/120992588-0ac11000-c783-11eb-8fdb-43404f52f97b.png)
<center>The above image shows the successful instantiation of triton server</center>

## Verify Triton is running correctly

``` curl -v http://localhost:8000/v2/health/ready ```

The expected output should be (by default triton provide services on port 8000) : <br/>
< HTTP/1.1 200 OK. <br/>
< Content-Length: 0 <br/>
< Content-Type: text/plain <br/>


# Part2: Setting up Triton Inference client
In this part we will download the libraries required to interact with triton server i.e sending inference requests (input data) to the deployed models and recieving back the predictions.
It is recommended to install the below packages in a separate [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
## Libraries required:
  1. pip install nvidia-pyindex
  2. pip install tritonclient[all]
  3. python -m pip install grpcio
  4. python -m pip install grpcio-tools
  5. pip install geventhttpclient
  6. pip install attrdict
  7. pip install Pillow

## Image classification Example:
Once the libraries are installed we can start communicating with triton server using inference scripts:

e.g. ``` python image_client.py -c 3  -m inception_graphdef -s INCEPTION path/to/example_image ```

## References:
1. https://medium.com/nvidia-ai/how-to-deploy-almost-any-hugging-face-model-on-nvidia-triton-inference-server-with-an-8ee7ec0e6fc4
2. https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us
3. https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md
