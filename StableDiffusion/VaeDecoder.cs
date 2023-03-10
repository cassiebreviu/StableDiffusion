﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace StableDiffusion
{
    public static class VaeDecoder
    {
        public static Tensor<float> Decoder(List<NamedOnnxValue> input, string vaeDecoderModelPath)
        {
            // Set CUDA EP
            var sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();

            // Create an InferenceSession from the Model Path.
            var vaeDecodeSession = new InferenceSession(vaeDecoderModelPath, sessionOptions);

            // Run session and send the input data in to get inference output. 
            var output = vaeDecodeSession.Run(input);
            var result = (output.First().Value as Tensor<float>);

            return result;
        }

        // create method to convert float array to an image with imagesharp
        public static Image<Rgba32> ConvertToImage(Tensor<float> output, int width, int height)
        {
            var result = new Image<Rgba32>(width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }
            return result;
        }
    }
}
