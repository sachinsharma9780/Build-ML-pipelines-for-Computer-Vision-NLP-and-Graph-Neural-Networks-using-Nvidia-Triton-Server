# AI-Enterprise-Workshop-Building-ML-Pipelines
In this workshop we are going to use [Nvidia's Triton Inference server](https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us) (formerly known as TensorRT Inference Server), which simplifies the deployment of AI models at scale in production. It natively supports multiple framework backends like TensorFlow, PyTorch, ONNX Runtime, Python, and even custom backends. It supports different types of inference queries through advanced batching and scheduling algorithms, supports live model updates, and runs models on CPUs and GPUs. Triton is also designed to increase inference performance by maximizing hardware utilization through concurrent model execution and dynamic batching. Concurrent execution allows you to run multiple copies of a model, and multiple different models, in parallel on the same GPU. Through dynamic batching, Triton can dynamically group inference requests on the server-side to maximize performance. For this examination, we focus on hosting/deploying multiple trained models (Tensorflow, PyTorch) on triton inference server leverage its full potential. Once models are deployed, we can make inference requests and can get back the predictions. 

<center> <img width="797" alt="Screenshot 2021-07-07 at 08 58 50" src="https://user-images.githubusercontent.com/40523048/124829341-cb9bff80-df78-11eb-99ef-9b650010b039.png"> </center>

 The above image represents the [Triton Inference Server Architecture](https://developer.nvidia.com/blog/simplifying-ai-inference-in-production-with-triton/) with its various supported components.

# Nvidia Triton Inference Server Features
![Screenshot 2021-09-13 at 11 42 45](https://user-images.githubusercontent.com/40523048/133061949-f49d636c-b2a4-4dc2-b80e-e32896d2ae64.png)


![1_126iG2mnfl4i6iH9FKu3sg](https://user-images.githubusercontent.com/40523048/120965914-c4a98380-c765-11eb-86f0-eb2ce2574e97.png)
Image depicting the capability of [Nvidia's Triton Inference server](https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us) to host Multiple heterogeneous deep learning frameworks on a GPU or a CPU (depending upon the backened).

For setting up the Triton inference server, we generally need to pass two hurdles: 1) Set up our own inference server, and 2) After that, we have to write a client-side python script that can communicate with the inference server to send requests (in our case text) and get back predictions or image/text feature embeddings.

# Part1: Setting up Triton Inference Server on the machine
Let's start by setting up a triton server locally on the computer by following the below steps.

### Quickstart with Docker
```
1. Install Docker
2. docker pull nvcr.io/nvidia/tritonserver:21.06.1-py3
3. git clone https://github.com/sachinsharma9780/AI-Enterprise-Workshop-Building-ML-Pipelines.git
4. cd ./AI-Enterprise-Workshop-Building-ML-Pipelines
5. docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:21.06.1-py3 tritonserver --model-repository=/models
6. curl -v http://localhost:8000/v2/health/ready

Continue to Part 2 below..
```

## Install Docker
[Docker](https://docs.docker.com/get-docker/)

## Pulling triton server docker image from Nvidia NGC:
1. [Download](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver/) docker image
2. or use the command: 
``` docker pull nvcr.io/nvidia/tritonserver:21.06.1-py3 ```
4. Image size: 10.6 GB (10-15 mins to install) 
5. To view the downloaded docker image: 
``` docker images ```

## Create a model repository to add your models:
1. Clone the [Triton Inference Server GitHub repository](https://github.com/triton-inference-server/server.git
) if you need an example model repository (this will also download some pre-trained models structured in a manner as expected by Triton)
2. After cloning, you can find the trained models under: server → docs →examples →model_repository
3. Or you can clone this repo, and in the model_repository folder, I have already stored some default trained models with their corresponding configuration file, which comes along while cloning the above repository.
4. Instantiate triton server using the cmd: </br>
``` docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/example/model/repository:/models docker image tritonserver --model-repository=/models ```

Note: Where docker image is nvcr.io/nvidia/tritonserver:<xx.yy>-py3 if you pulled the Triton container from NGC. -v flag points to the path of your model repository where all your models are stored, and --gpus=1 flag refers to 1 system GPU should be available to Triton for inference as shown above.

e.g. ``` docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Users/sachin/Desktop/arangodb/scripts/triton/model_repository:/models nvcr.io/nvidia/tritonserver:21.06.1-py3 tritonserver --model-repository=/models ```


![Screenshot 2021-06-07 at 11 24 57](https://user-images.githubusercontent.com/40523048/120992588-0ac11000-c783-11eb-8fdb-43404f52f97b.png)
<center>The above image shows the successful instantiation of triton server</center>

## Verify Triton is running correctly

``` curl -v http://localhost:8000/v2/health/ready ```

The expected output should be (by default triton provide services on port 8000): <br/>
< HTTP/1.1 200 OK. <br/>
< Content-Length: 0 <br/>
< Content-Type: text/plain <br/>


# Part2: Setting up Triton Inference client

In this part, we will download the libraries required to interact with triton server, i.e., sending inference requests (input data) to the deployed models and receiving back the predictions.
It is recommended to install the below packages in a separate [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

## Install Required Libraries
1. `cd` into `scripts` folder
2. `pip install -r .\requirements.txt` OR install as show below

## Install Libraries Individually:
  1. pip install nvidia-pyindex
  2. pip install tritonclient[all]
  3. pip install torch
  4. pip install transformers
  5. python -m pip install grpcio
  6. python -m pip install grpcio-tools
  7. pip install geventhttpclient
  8. pip install attrdict
  9. pip install Pillow

## Order of Execution

### Application #1: Deploying Hugging Face transformer model on Triton Inference server with an application to Zero Shot Text Classiifcation
1) Start with creating a Triton acceptabele model using a notebook (under folder create_triton_acceptable_models) trace_pytorch_models.ipynb
2) Add this created model into a model repository
3) Start the Triton server with this newly added model
4) Run the application using the notebook triton_client_zero_shot_text_classification_application.ipynb

### Application #2: Movie recommendation with Triton inference server and ArangoDB:
1) Start with creating a Triton acceptabele model using a notebook trace_sentence_repn_bert_model.ipynb
2) Add this created model into a model repository
3) Start the Triton server with this newly added model (you can add multiple models in this repository depending upon the memory)
4) Run the application using the notebook movie_recommendation_triton_client.ipynb

### Application #3: Graph ML, Nvidia Triton, and ArangoDB: Amazon Product Recommendation (APR) Application
1) Train GrapSage model on APR dataset using a notebook Comprehensive_GraphSage_Guide_with_PyTorchGeometric.ipynb
2) Load APR graph dataset into arangodb using the [dump](https://drive.google.com/drive/folders/1JF0gkAMmSlrsmmnB9uzeZdgmX8NgFwV4) and [arangorestore](https://www.arangodb.com/docs/stable/programs-arangorestore.html) utility. 
For eg. ``` arangorestore --input-directory "dump" ```
4) Either you can chose your own generated checkpoints from 1) or I have already stored them under chechkpoint folder for both GPU and CPU trained GraphSage model.
5) Create a trace on GraphSage model using these checkpoints using notebook trace_obgn-product_graphsage_model.ipynb
6) Update your model_repository for this traced model like shown in model_repository folder (graph_embeddings)
7) Start the Triton server with this newly added model (you can add multiple models in this repository depending upon the memory)
8) Run the application using the graph_ml_triton_arangodb_product_recommendation_app.ipynb

## Dump Folder
This folder already contains movie embeddings for all the movie descriptions present inside the imdb dataset. We did this to save time in case you run the movie recommendation notebook on CPU then it takes some time to generate movie embeddings and then store them in ArangoDB. In order to restore the movie embeddings inside the ArangoDb we can use its [arangorestore](https://www.arangodb.com/docs/stable/programs-arangorestore.html) utiliy.

## Image classification Example:
Once the libraries are installed, we can start communicating with triton server using inference scripts:

e.g. ``` python image_client.py -c 3  -m inception_graphdef -s INCEPTION path/to/example_image ```

## Slide Deck
[Presentation](https://docs.google.com/presentation/d/1W0BnEsJrN5tR1E7ahVZuE70NnksIwHmROxieh5rIa2w/edit#slide=id.ge266904e26_0_530)

## WorkShop-1 Link
[YouTube](https://www.youtube.com/watch?v=vOIm7Hibgdo&t=1952s)

## References:
1. https://medium.com/nvidia-ai/how-to-deploy-almost-any-hugging-face-model-on-nvidia-triton-inference-server-with-an-8ee7ec0e6fc4
2. https://developer.nvidia.com/nvidia-triton-inference-server?ncid=partn-88872#cid=dl13_partn_en-us
3. https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md
