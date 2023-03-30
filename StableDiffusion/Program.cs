using Microsoft.ML.OnnxRuntime.Tensors;
using System.Net.Http.Headers;
using System.Net.Http.Json;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Inference local or on Azure OAI Dall-E
            var isLocal = true;

            //Default args
            var prompt = "a fireplace in an old cabin in the woods";
            Console.WriteLine(prompt);

            if (isLocal)
            {

                //test how long this takes to execute
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Directory.SetCurrentDirectory(@"..\..\..\..");



                // Number of denoising steps
                var num_inference_steps = 15;
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
                if (image == null)
                {
                    Console.WriteLine("Unable to create image, please try again.");
                }

                // Stop the timer
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Time taken: " + elapsedMs + "ms");
            }
            else
            {   
                // Update the url and api key from the Azure Open AI Resource. This is not Prod ready and is meant to be used as a proof of concept.
                string url = "";
                string api_key = "";
                HttpClient client = new HttpClient();
                client.BaseAddress = new Uri(url);
                client.DefaultRequestHeaders.Add("api-key", api_key);
                client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                var body = new
                {
                    caption = prompt,
                    resolution = "1024x1024"
                };
                var response = client.PostAsJsonAsync(url, body).Result;
                var operation_location = response.Headers.GetValues("Operation-Location").FirstOrDefault();
                var retry_after = response.Headers.GetValues("Retry-after").FirstOrDefault();
                var azureOpenAiResult = new AzureOpenAiResult();
                azureOpenAiResult.status = "";
                while (azureOpenAiResult.status != "Succeeded")
                {
                    System.Threading.Thread.Sleep(int.Parse(retry_after) * 1000);
                    response = client.GetAsync(operation_location).Result;
                    azureOpenAiResult = response.Content.ReadFromJsonAsync<AzureOpenAiResult>().Result;
                }
                // This is a link to the image that was created. Click the link to view the image.
                Console.WriteLine(azureOpenAiResult.result.contentUrl);

            }

        }

    }
}