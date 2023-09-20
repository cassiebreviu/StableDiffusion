using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class SafetyChecker
    {
        public static bool IsNotSafe(Tensor<float> resultImage, StableDiffusionConfig config)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(resultImage, config);
            //images input
            var inputImagesTensor = ReorderTensor(inputTensor);

            var input = new List<NamedOnnxValue>
            {
                //batch channel height width
                 NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),

                 //batch, height, width, channel
                 NamedOnnxValue.CreateFromTensor("images", inputImagesTensor)
            };

            // Run session and send the input data in to get inference output. 
            using (var sessionOptions = config.GetSessionOptionsForEp())
            using (var session = new InferenceSession(config.SafetyModelPath, sessionOptions))
            using (var output = session.Run(input))
            {
                var result = output.LastElementAs<IEnumerable<bool>>();
                return result.First();
            }
        }

        private static DenseTensor<float> ReorderTensor(Tensor<float> inputTensor)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }

            return inputImagesTensor;
        }
        private static DenseTensor<float> ClipImageFeatureExtractor(Tensor<float> imageTensor, StableDiffusionConfig config)
        {
            // Read image
            //using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            //convert tensor result to image
            var image = new Image<Rgba32>(config.Width, config.Height);

            for (var y = 0; y < config.Height; y++)
            {
                for (var x = 0; x < config.Width; x++)
                {
                    image[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgba32> pixelSpan = image.GetPixelRowSpan(y);

                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;

        }
    }
}

