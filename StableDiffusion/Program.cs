using Microsoft.ML.OnnxRuntime.Tensors;
using static StableDiffusion.StableDiffusionConfig;

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

            var config = new StableDiffusionConfig
                {
                // Number of images requested.
                BatchSize = 1,
                // Number of denoising steps.
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance.
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 0
                };

            // For some editors the dynamic path doesnt work. If you need to change to a static path
            // update the useStaticPath to true and update the paths below.
            config.SetModelPaths();


            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TextProcessing.TokenizeText(prompt, config);
            var textPromptEmbeddings = TextProcessing.TextEncoder(textTokenized, config).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = TextProcessing.CreateUncondInput();
            var uncondEmbedding = TextProcessing.TextEncoder(uncondInputTokens, config).ToArray();

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            }

            // Inference Stable Diff
            var image = UNet.Inference(textEmbeddings, config);

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