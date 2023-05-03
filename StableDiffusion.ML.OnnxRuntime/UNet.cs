using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class UNet
    {
        public static List<NamedOnnxValue> CreateUnetModelInput(
Tensor<float> encoderHiddenStates, Tensor<float> sample, float timeStep)
        {
            // convert encoderHiddenStates to float16 tensors
            var encoderHiddenStatesFloat16 = TensorHelper.ConvertFloatToFloat16(encoderHiddenStates);
            var sampleFloat16 = TensorHelper.ConvertFloatToFloat16(sample);

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor<Float16>("encoder_hidden_states", encoderHiddenStatesFloat16),
                NamedOnnxValue.CreateFromTensor<Float16>("sample", sampleFloat16),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<Float16>(new Float16[] { BitConverter.HalfToUInt16Bits((Half)timeStep) }, new int[] { 1 }))
            };
            
            return input;

        }

        public static Tensor<float> GenerateLatentSample(StableDiffusionConfig config, int seed, float initNoiseSigma)
        {
            return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
        }
        public static Tensor<float> GenerateLatentSample(int height, int width, int seed, float initNoiseSigma)
        {
            var random = new Random(seed);
            var batchSize = 1;
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = (float)(standardNormalRand * initNoiseSigma);
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }

        private static Tensor<float> performGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                {
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                    {
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                        {
                            noisePred[i, j, k, l] = (float)(noisePred[i, j, k, l] +(float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]));
                        }
                    }
                }
            }
            return noisePred;
        }

        public static SixLabors.ImageSharp.Image Inference(String prompt, StableDiffusionConfig config)
        {
            // Preprocess text
            var textEmbeddings = TextProcessing.PreprocessText(prompt, config);

            var scheduler = new LMSDiscreteScheduler();
            //var scheduler = new EulerAncestralDiscreteScheduler();
            var timesteps = scheduler.SetTimesteps(config.NumInferenceSteps);
            //  If you use the same seed, you will get the same image result.
            var seed = new Random().Next();
            //var seed = 329922609;
            Console.WriteLine($"Seed generated: {seed}");
            // create latent tensor

            var latents = GenerateLatentSample(config, seed, scheduler.InitNoiseSigma);

            var sessionOptions = config.GetSessionOptionsForEp();
            // Create Inference Session
            var unetSession = new InferenceSession(config.UnetOnnxPath, sessionOptions);

            var input = new List<NamedOnnxValue>();
            for (int t = 0; t < timesteps.Length; t++)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), new[] { 2, 4, config.Height / 8, config.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

                Console.WriteLine($"scaled model input {latentModelInput[0]} at step {t}.");
                input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);

                // Run Inference
                var output = unetSession.Run(input);
                var outputTensor = (output.ToList().First().Value as DenseTensor<Float16>);

                // Split tensors from 2,4,64,64 to 1,4,64,64
                var splitTensors = TensorHelper.SplitTensor(outputTensor, new[] { 1, 4, config.Height / 8, config.Width / 8 });
                var noisePred = splitTensors.Item1;
                var noisePredText = splitTensors.Item2;

                // Perform guidance
                noisePred = performGuidance(noisePred, noisePredText, config.GuidanceScale);

                // LMS Scheduler Step
                latents = scheduler.Step(noisePred, timesteps[t], latents);
                Console.WriteLine($"latents result after step {t} min {latents.Min()} max {latents.Max()}");

            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents.ToArray(), (float)(1.0f / 0.18215f), latents.Dimensions.ToArray());
            var latentsFloat16 = TensorHelper.ConvertFloatToFloat16(latents);
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latentsFloat16) };

            // Decode image
            var imageResultTensor = VaeDecoder.Decoder(decoderInput, config);

            // TODO: Fix safety checker model
            //var isSafe = SafetyChecker.IsSafe(imageResultTensor);

            ////if (isSafe == 1)
            //{ 
            var image = VaeDecoder.ConvertToImage(imageResultTensor, config);
            return image;
            //}

        }

    }
}
