using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            //test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();
            Directory.SetCurrentDirectory(@"..\..\..");

            //Default args
            var prompt = "a cat fashion show with sunglasses";
            Console.WriteLine(prompt);

            // Number of denoising steps
            var num_inference_steps = 10;
            // Scale for classifier-free guidance
            var guidance_scale = 7.5;
            //num of images requested
            var batch_size = 1;

            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TextProcessing.TokenizeText(prompt);
            var textPromptEmbeddings = TextProcessing.TextEncoder(textTokenized).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = TextProcessing.CreateUncondInput();
            var uncondEmbedding = TextProcessing.TextEncoder(uncondInputTokens).ToArray();

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
            var image = UNet.Inference(num_inference_steps, textEmbeddings, guidance_scale, batch_size, height, width);

            // If image failed or was unsafe it will return null.
            if( image == null )
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