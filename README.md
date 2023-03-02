
# Inference Stable Diffusion with C# and ONNX Runtime

This repo contains the logic to do inferencing for the popular Stable Diffusion deep learning model in C#. Stable Diffusion models denoise a static image to create an image that represents the text prompt given by the user.

For example the sentence "make a picture of green tree with flowers around it and a red sky" is created as a text embedding from the [CLIP model](https://huggingface.co/docs/transformers/model_doc/clip) that "understand" text and image relationship. A random noise image based on the seed number is created and then denoised to create an image that represents the text prompt.

```text
"make a picture of green tree with flowers around it and a red sky" 
```
| Latent Seed Image | Resulting image |
| :--- | :--- |
<img src="images/latent.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="images/sample-output-stablediff.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

## Prerequisites
This tutorial can be run locally or in the cloud by leveraging Azure Machine Learning compute.

- [Download the Source Code from GitHub](https://github.com/cassiebreviu/StableDiffusion)

To run locally:

- [Visual Studio](https://visualstudio.microsoft.com/downloads/) or [VS Code](https://code.visualstudio.com/Download)
- A GPU enabled machine with CUDA EP Configured. This was built on a GTX 3070 and it has not been tested on anything smaller. Follow [this tutorial to configure CUDA and cuDNN for GPU with ONNX Runtime and C# on Windows 11](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)

To run in the cloud with Azure Machine Learning:

- [Azure Subscription](https://azure.microsoft.com/free/)
- [Azure Machine Learning Resource](https://azure.microsoft.com/services/machine-learning/)

## Use Hugging Face to download the Stable Diffusion models

Download the [ONNX Stable Diffusion models from Hugging Face](https://huggingface.co/models?sort=downloads&search=Stable+Diffusion).

 - [ONNX Models for v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/onnx)
 - [ONNX Models for v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)


Once you have selected a model version repo, click `Files and Versions`, then select the `ONNX` branch. If there isn't an ONNX model branch available, use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial for PyTorch](https://learn.microsoft.com/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

- Clone the repo:
```text
git lfs install
git clone https://huggingface.co/<contributor>/<model-name>
```

- Copy the folders with the ONNX files to the C# project folder `\StableDiffusion\StableDiffusion`. The folders to copy are: `unet`, `vae_decoder`, `text_encoder`, `safety_checker`.

## Resources
- [Stable Diffusion C# Tutorial for the Repo]()
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable-diffusion)
