using StableDiffusion.ML.OnnxRuntime;

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
                //num of images requested
                BatchSize = 1,
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 0,
                // Update paths to your models
                TokenizerOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\text_tokenizer\custom_op_cliptok.onnx",
                TextEncoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\text_encoder\model.onnx",
                UnetOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\unet\model.onnx",
                VaeDecoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\vae_decoder\model.onnx",
                SafetyModelPath = @"C:\code\StableDiffusion\StableDiffusion\models\safety_checker\model.onnx",
                OutputImagePath = "sample.png"
            };

            // Inference Stable Diff
            var image = UNet.Inference(prompt, config);

            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }
            else
            {
                // Get exectiion path and append output image name
                var path = Path.Combine(Directory.GetCurrentDirectory(), config.OutputImagePath);
                Console.WriteLine("Image saved to: " + path);
            }

            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}