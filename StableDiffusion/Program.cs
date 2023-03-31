using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            //test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();

            //Default args
            var prompt = "a fireplace in an old cabin in the woods";
            Console.WriteLine(prompt);

            // Number of denoising steps
            var num_inference_steps = 15;
            // Scale for classifier-free guidance
            var guidance_scale = 7.5;
            //num of images requested
            var batch_size = 1;

            // For some editors the dynamic path doesnt work. If you need to change to a static path
            // update the useStaticPath to true and update the paths below.
            var useStaticPath = false;
            var OrtExtensionsPath = "";
            var TokenizerOnnxPath = "";
            var TextEncoderOnnxPath = "";
            var UnetOnnxPath = "";
            var VaeDecoderOnnxPath = "";


            if (!useStaticPath)
            {
                Directory.SetCurrentDirectory(@"..\..\..\..");
                OrtExtensionsPath = Directory.GetCurrentDirectory().ToString() + ("\\ortextensions.dll");
                TokenizerOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_tokenizer\\custom_op_cliptok.onnx");
                TextEncoderOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_encoder\\model.onnx");
                UnetOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\unet\\model.onnx");
                VaeDecoderOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\vae_decoder\\model.onnx");
            }
            else
            {
                OrtExtensionsPath = "ortextensions.dll";
                TokenizerOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\text_tokenizer\custom_op_cliptok.onnx";
                TextEncoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\text_encoder\model.onnx";
                UnetOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\unet\model.onnx";
                VaeDecoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\vae_decoder\model.onnx";
            }
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TextProcessing.TokenizeText(prompt, OrtExtensionsPath, TokenizerOnnxPath);
            var textPromptEmbeddings = TextProcessing.TextEncoder(textTokenized, TextEncoderOnnxPath).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = TextProcessing.CreateUncondInput();
            var uncondEmbedding = TextProcessing.TextEncoder(uncondInputTokens, TextEncoderOnnxPath).ToArray();

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            }

            var height = 512;
            var width = 512;

            // Inference Stable Diff
            var image = UNet.Inference(num_inference_steps, textEmbeddings, guidance_scale, batch_size, UnetOnnxPath, VaeDecoderOnnxPath, height, width);

            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }

            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}