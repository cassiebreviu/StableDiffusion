using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace StableDiffusion
{
    public class Program
    {
        const string BasePath = "../../../";
        const string OrtExtensionsPath = BasePath + "ortextensions.dll";
        const string OnnxModelsBasePath = BasePath; // Instead of copying models point to stable diffusion repo e.g. `stable-diffusion-v1-5`
        const string TokenizerOnnxPath = OnnxModelsBasePath + "text_tokenizer/custom_op_cliptok.onnx";
        const string TextEncoderOnnxPath = OnnxModelsBasePath + "text_encoder/model.onnx";
        const string UnetOnnxPath = OnnxModelsBasePath + "unet/model.onnx";
        const string VaeDecoderOnnxPath = OnnxModelsBasePath + "vae_decoder/model.onnx";

        static void Main(string[] args)
        {
            // Test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();

            // Default args
            var prompt = "a fireplace in an old cabin in the woods";
            Console.WriteLine(prompt);

            // Number of denoising steps
            var num_inference_steps = 40;
            // Scale for classifier-free guidance
            var guidance_scale = 7.5;
            //num of images requested
            var batch_size = 1;

            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TextProcessing.TokenizeText(prompt, OrtExtensionsPath, TokenizerOnnxPath);
            var textPromptEmbeddings = TextProcessing.TextEncoder(textTokenized, TextEncoderOnnxPath).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = TextProcessing.CreateUncondInput();
            var uncondEmbedding = TextProcessing.TextEncoder(uncondInputTokens, TextEncoderOnnxPath).ToArray();

            // Concant textEmeddings and uncondEmbedding
            const int embeddingLength = 768;
            var textEmbeddings = new DenseTensor<float>(new[] { 2, 77, embeddingLength });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / embeddingLength, i % embeddingLength] = uncondEmbedding[i];
                textEmbeddings[1, i / embeddingLength, i % embeddingLength] = textPromptEmbeddings[i];
            }

            var height = 256;
            var width = 256;

            // Inference Stable Diff
            var image = UNet.Inference(UnetOnnxPath, VaeDecoderOnnxPath,
                num_inference_steps, textEmbeddings, guidance_scale, batch_size, height, width);

            image.Save("Sample.png");

            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");
        }
    }
}