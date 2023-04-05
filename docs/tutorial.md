# How to run ChatRWKV step-by-step guide

In this tutorial, we will guide you through the process of running ChatRWKV, a conversational interface to the RWKV models. We will cover the steps necessary to download the source code, create a virtual environment, install requisites, and launch ChatV2. Additionally, we'll show you how to customize ChatV2 to get the best experience possible.

## Prerequisites

Ensure you have the following software installed on your computer:

- Python 3.8, 3.9 or 3.10
- Git

## Download the source code

First, open a terminal and move to your prefered directory (in this example, we will be using dev directory under our home directory):

```sh
cd ~/dev
git clone https://github.com/BlinkDL/ChatRWKV.git
```

This will clone the ChatRWKV repository to your local machine.

## Create a virtual environment

To create a virtual environment, move to the ChatRWKV directory and run the following commands:

```sh
cd ChatRWKV/
python -m venv ./.venv
source ./.venv/bin/activate
```

## Install requisites

To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

This will install all required libraries to run ChatRWKV v2.

## Download and set the model
Visit Hugging Face and download the model of your choice. You can test some of these models before downloading them locally:

- [RWKV-4-Pile-14B-20230313-ctx8192-test1050](https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio): trained on the Pile, with a context of 8192 tokens. Not optimized for conversations.
- [Raven - RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096](https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B): trained on Alpaca datasets and more, with a context size of 4096 tokens. Optimized for instruction based interactions.

Make sure to check the prompt examples at the bottom of each page, so you can get used to the prompting style required for each model. Please note, that only the Raven model has been tuned for chat conversations, so other models in the RWKV-4 family will not return proper answers when asked in a conversational style.

Once you know what model would you like to test, open the "Files and versions" tab at the top. Newer models appear frequently at the bottom of the list. Select the model file and click on the name (checking the different languages and versions). A new page will open with a link that says "Download". Right click on the link and save it to your prefered destination under models.

Tip: For better organization, create a folder called models and subfolders for each model. Then, place the model files within the appropriate subfolder.

## Customizing ChatV2

Before launching ChatV2, we need to customize it to match our machine configuration. To do so, navigate to the 'v2' folder and edit chat.py with your favorite text editor.

```sh
cd v2/
emacs chat.py
```

Bellow you will find some of the most important parameters that you need to configure to get the best experience out of ChatRWKV.

### Model name

Once we have downloaded our model, we need to customize chat.py to point to the actual file. Open your favorite text editor and replace the model in the code with the one you want to use. In this example, we will use:

```python
if CHAT_LANG == 'English':
    args.MODEL_NAME = './models/rwkv-4-raven/RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096.pth' # try +i for "Alpaca instruct"
```

### Strategy

Depending on your GPU/CPU configuration you will need to set a strategy for the execution. GPUs are way faster than CPU, so the overall goal is to fit as many layers as possible in the GPU.

Check the following table for an overview of the different models (using args.ctx_len = 1024):

| Model                                  | Number of layers | Strategy | Average VRAM in GPU per layer | Total VRAM required |
|----------------------------------------|------------------|----------|-------------------------------|---------------------|
| RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096 |               33 |   fp16i8 |                270 MB aprox. |            9825 MB |
| RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth |         41 |   fp16i8 |                 354 MB aprox. |           14500 MB |

As a rule of thumb, fp16i8 will require 50% less memory that fp16. Given our goal is to load the model to the GPU memory, we could fit the 33 layers of the Raven-7B in a card with 12 Gb of VRAM (like RTX2060 or RTX3060). To do so, we should configure the following strategy:

```python
args.strategy = 'cuda fp16i8'
```

To load the 14B version, we need to offload some of the layers to the CPU/main memory using the following strategy parameter:
```python
args.strategy = 'cuda fp16i8 *28 -> cpu fp32'
```

If we had a RTX3090 with 24 GB, it would not be necessary to offload any layer. However, if we wanted to load the full fp16 weights, we should then offload some layers to the CPU (or to a second GPU). To calculate the number of layers required (assuming fp16 takes double the VRAM of the fp16i8), let's multiply the number of layers 41 per the VRAM per layer 710 MB, to get a total of 29110 MB (around 29 GB of VRAM). 

To fit the model in a 24 Gb VRAM card, we need to substract 1 Gb (to the base operating system) and then divide the remaing VRAM by the layer size: (23 * 1024)  / 710 = 23552 / 710 = 33,17 layers. This would be the required strategy to load the model in fp16:

```python
args.strategy = 'cuda fp16i8 *33 -> cpu fp32'
```

For extended information about the strategies, please check [rwkv pip repository](https://pypi.org/project/rwkv/)

### Enable CUDA

Setting RWKV_CUDA_ON to 1 will compile a CUDA kernel, making the model run faster (10x faster according to the comments). This is the best performance improvement you can do to get faster results, by replacing a single digit.

Make sure your chat.py includes this line:

```python
os.environ["RWKV_CUDA_ON"] = '1' 
```

### Choose a predefined conversation style

ChatRWKV can be started with several predefined conversations. Choose one from the list provided in the chat.py comments.

```python
# -1.py for [User & Bot] (Q&A) prompt
# -2.py for [Bob & Alice] (chat) prompt
# -3.py for a very long (but great) chat prompt (requires ctx8192, and set RWKV_CUDA_ON = 1 or it will be very slow)
```

Test all different options and choose the one that suits your usage better. For example, this option would launch a Q&A interface.

```python
PROMPT_FILE = f'{current_path}/prompt/default/{CHAT_LANG}-1.py'
```

## Launching ChatV2

Now that everything is ready, you can execute the chat.py script under the v2 directory:

```sh
cd v2/
python chat.py
```

Enjoy your time talking to RWKV models!