
# Inference Stable Diffusion with C# and ONNX Runtime

This package contains the logic to do inferencing for the popular Stable Diffusion deep learning model in C#.  Stable Diffusion models take a text prompt and create an image that represents the text.


# How to use this NuGet package

- Download the ONNX Stable Diffusion models from Hugging Face
    - [Stable Diffusion Models v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/onnx)
    - [Stable Diffusion Models v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)
    
- Once you have selected a model version repo, click `Files and Versions`, then select the `ONNX` branch. If there isn't an ONNX model branch available, use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial for PyTorch](https://learn.microsoft.com/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

- Clone the model repo:
```text
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 -b onnx
```

- Copy the folders with the ONNX files to the C# project folder `models`. The folders to copy are: `unet`, `vae_decoder`, `text_encoder`, `safety_checker`.

- Install the following NuGets for DirectML
```xml
<PackageReference Include="Microsoft.ML" Version="2.0.1" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.14.1" />
```
- Cuda support coming soon.

- Sample logic for implementing in your project
```csharp
    //Default args
    var prompt = "a fireplace in an old cabin in the woods";
    Console.WriteLine(prompt);

    var config = new StableDiffusionConfig
    {
        // Number of denoising steps
        NumInferenceSteps = 15,
        // Scale for classifier-free guidance
        GuidanceScale = 7.5,
        // Set your preferred Execution Provider. Currently DirectML and CPU are supported.
        ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.DirectML,
        // Set GPU Device ID.
        DeviceId = 1,
        // Update paths to your models
        TextEncoderOnnxPath = @".\models\text_encoder\model.onnx",
        UnetOnnxPath = @".\models\unet\model.onnx",
        VaeDecoderOnnxPath = @".\models\vae_decoder\model.onnx",
        SafetyModelPath = @".\models\safety_checker\model.onnx",
    };

    // Inference Stable Diff
    var image = UNet.Inference(prompt, config);

    // If image failed or was unsafe it will return null.
    if (image == null)
    {
        Console.WriteLine("Unable to create image, please try again.");
    }

```

- Set Build for x64 

- Hit `F5` to run the project in Visual Studio or `dotnet run` in the terminal to run the project in VS Code.

__________________________
##  Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
- [Stable Diffusion C# Tutorial for this Repo](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)
