
# Inference Stable Diffusion with C# and ONNX Runtime

This repo contains the logic to do inferencing for the popular Stable Diffusion deep learning model in C#.  Stable Diffusion models take a text prompt and create an image that represents the text.

For the below example sentence the [CLIP model](https://huggingface.co/docs/transformers/model_doc/clip) creates a text embedding that connects text to image. A random noise image is created and then denoised with the `unet` model and scheduler algorithm to create an image that represents the text prompt. Lastly the decoder model `vae_decoder` is used to create a final image that is the result of the text prompt and the latent image.

```text
"make a picture of green tree with flowers around it and a red sky" 
```
| Auto Generated Random Latent Seed Input | Resulting image output|
| :--- | :--- |
<img src="images/latent.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="images/sample-output-stablediff.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

## More Images Created with this Repo:

| <img src="images/cat-sunglasses.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="images/dog-beach-sample.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

| <img src="images/cabin.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="images/shipwreck.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

## Prerequisites

- [Visual Studio](https://visualstudio.microsoft.com/downloads/) or [VS Code](https://code.visualstudio.com/Download)
- A GPU enabled machine with CUDA or DirectML on Windows
    - Configure CUDA EP.  Follow [this tutorial to configure CUDA and cuDNN for GPU with ONNX Runtime and C# on Windows 11](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)
    - Windows comes with DirectML support. No additional configuration is needed. Be sure to clone the [`direct-ML-EP`](https://github.com/cassiebreviu/StableDiffusion/tree/direct-ML-EP) branch of this repo if you choose this option.
    - This was built on a GTX 3070 and it has not been tested on anything smaller.
- Clone this repo
```git
git clone https://github.com/cassiebreviu/StableDiffusion.git
```

## Use Hugging Face to download the Stable Diffusion models

Download the [ONNX Stable Diffusion models from Hugging Face](https://huggingface.co/models?sort=downloads&search=Stable+Diffusion).

 - [Stable Diffusion Models v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/onnx)
 - [Stable Diffusion Models v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)


Once you have selected a model version repo, click `Files and Versions`, then select the `ONNX` branch. If there isn't an ONNX model branch available, use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial for PyTorch](https://learn.microsoft.com/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

- Clone the model repo:
```text
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 -b onnx
```

- Copy the folders with the ONNX files to the C# project folder `\StableDiffusion\StableDiffusion`. The folders to copy are: `unet`, `vae_decoder`, `text_encoder`, `safety_checker`.

- Set Build for x64 

- Hit `F5` to run the project in Visual Studio or `dotnet run` in the terminal to run the project in VS Code.

_____________________
## Follow the full Stable Diffusion C# Tutorial for this Repo [here](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)

__________________________
##  Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
- [Stable Diffusion C# Tutorial for this Repo](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)
