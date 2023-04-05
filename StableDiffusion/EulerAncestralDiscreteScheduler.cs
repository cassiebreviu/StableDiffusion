using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion
{
    public class EulerAncestralDiscreteScheduler : SchedulerBase
    {
        private readonly string prediction_Type;
        public override float InitNoiseSigma { get; set; }
        public int num_inference_steps;
        public override List<int> Timesteps { get; set; }
        public override Tensor<float> Sigmas { get; set; }

        //[RegisterToConfig]
        public EulerAncestralDiscreteScheduler(
            int num_train_timesteps = 1000,
            float beta_start = 0.00085f,
            float beta_end = 0.012f,
            string beta_schedule = "scaled_linear",
            List<float> trained_betas = null,
            string prediction_type = "epsilon"
        ) : base(num_train_timesteps)
        {
            var alphas = new List<float>();
            var betas = new List<float>();
            prediction_Type = prediction_type;

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start = (float)Math.Sqrt(beta_start);
                var end = (float)Math.Sqrt(beta_end);
                betas = np.linspace(start, end, num_train_timesteps).ToArray<float>().Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            alphas = betas.Select(beta => 1 - beta).ToList();

            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma = (float)sigmas.Max();
        }



        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.InitNoiseSigma = (float)sigmas.Max();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] = (float)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public override DenseTensor<float> Step(Tensor<float> modelOutput,
               int timestep,
               Tensor<float> sample,
               int order = 4)
        {
            //if (timestep is int || timestep is NDArray || timestep is long)
            //{
            //    throw new ArgumentException(
            //        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to " +
            //        "`EulerDiscreteScheduler.step()` is not supported. Make sure to pass " +
            //        "one of the `scheduler.timesteps` as a timestep."
            //    );
            //}

            if (!this.is_scale_input_called)
            {
                Console.WriteLine(
                    "The `scale_model_input` function should be called before `step` to ensure correct denoising. " +
                    "See `StableDiffusionPipeline` for a usage example."
                );
            }

            //if (timestep is NDArray)
            //{
            //    timestep = timestep.ToArray<float>()[0];
            //}

            int stepIndex = this.Timesteps.IndexOf((int)timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample = null;
            if (this.prediction_Type == "epsilon")
            {
                //  pred_original_sample = sample - sigma * model_output
                var sigmaProduct = TensorHelper.MultipleTensorByFloat(modelOutput, sigma);
                predOriginalSample = TensorHelper.SubtractTensors(sample, sigmaProduct);
            }
            else if (this.prediction_Type == "v_prediction")
            {
                // * c_out + input * c_skip
                //predOriginalSample = modelOutput * (-sigma / Math.Pow(sigma * sigma + 1, 0.5)) + (sample / (sigma * sigma + 1));
                throw new NotImplementedException($"prediction_type not implemented yet: {prediction_Type}");
            }
            else if (this.prediction_Type == "sample")
            {
                throw new NotImplementedException($"prediction_type not implemented yet: {prediction_Type}");
            }
            else
            {
                throw new ArgumentException(
                    $"prediction_type given as {this.prediction_Type} must be one of `epsilon`, or `v_prediction`"
                );
            }

            float sigmaFrom = this.Sigmas[stepIndex];
            float sigmaTo = this.Sigmas[stepIndex + 1];

            // sigma_up = (sigma_to**2 * (sigma_from * *2 - sigma_to * *2) / sigma_from * *2) * *0.5

            var sigmaFromLessSigmaTo = (MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2));
            var sigmaUpResult = (MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo) / MathF.Pow(sigmaFrom, 2);

            var sigmaUp = 0f;

            // handle result if negative
            if (sigmaUpResult < 0)
            {
                sigmaUpResult = MathF.Abs(sigmaUpResult);
                sigmaUp = -MathF.Pow(sigmaUpResult, 0.5f);
            }
            else
            {
                sigmaUp = MathF.Pow(sigmaUpResult, 0.5f);
            }
            var sigmaDown = 0f;
            var sigmaDownResult = (MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2));
            // handle result if negative
            if (sigmaDownResult < 0)
            {
                sigmaDownResult = MathF.Abs(sigmaDownResult);
                sigmaDown = -MathF.Pow(sigmaDownResult, 0.5f);
            }
            else
            {
                sigmaDown = MathF.Pow(sigmaDownResult, 0.5f);
            }

            // 2. Convert to an ODE derivative
            var sampleMinusPredOriginalSample = TensorHelper.SubtractTensors(sample, predOriginalSample);
            DenseTensor<float> derivative = TensorHelper.DivideTensorByFloat(sampleMinusPredOriginalSample.ToArray(), sigma, predOriginalSample.Dimensions.ToArray());// (sample - predOriginalSample) / sigma;

            float dt = sigmaDown - sigma;

            var DerivativeDtProduct = TensorHelper.MultipleTensorByFloat(derivative, dt);
            DenseTensor<float> prevSample = TensorHelper.AddTensors(sample, DerivativeDtProduct);// sample + derivative * dt;

            //var noise = generator == null ? np.random.randn(modelOutput.shape) : np.random.RandomState(generator).randn(modelOutput.shape);
            var noise = GetRandomTensor(prevSample.Dimensions);

            var noiseSigmaUpProduct = TensorHelper.MultipleTensorByFloat(noise, (float)sigmaUp);
            prevSample = TensorHelper.AddTensors(prevSample, noiseSigmaUpProduct);// prevSample + noise * sigmaUp;

            //if (!returnDict)
            //{
            //    return prevSample;
            //}

            return prevSample;
        }


        public static NDArray BetasForAlphaBar(int numDiffusionTimesteps, float maxBeta = 0.999f)
        {
            float AlphaBar(int timeStep)
            {
                return (float)MathF.Pow(MathF.Cos((timeStep + 0.008f) / 1.008f * MathF.PI / 2), 2);
            }

            var betas = new List<float>();
            for (int i = 0; i < numDiffusionTimesteps; i++)
            {
                float t1 = (float)i / numDiffusionTimesteps;
                float t2 = (float)(i + 1) / numDiffusionTimesteps;
                betas.Add(Math.Min(1 - AlphaBar((int)t2) / AlphaBar((int)t1), maxBeta));
            }
            return np.array(betas).astype(np.float32);
        }

        public Tensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            var random = new Random();
            var latents = new DenseTensor<float>(dimensions);
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
                //latentsArray[i] = (float)standardNormalRand * this.InitNoiseSigma;
                latentsArray[i] = (float)standardNormalRand;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }
    }
}
