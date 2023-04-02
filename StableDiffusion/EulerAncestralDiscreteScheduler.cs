using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion
{
    public class EulerAncestralDiscreteScheduler : SchedulerBase
    {
        public static readonly int order = 1;
        private readonly string prediction_Type;
        public NDArray betas, alphas, sigmas;
        public override float InitNoiseSigma { get; set; }
        public int num_inference_steps;
        public override List<int> Timesteps { get; set; }
        public override Tensor<float> Sigmas { get; set; }

        //[RegisterToConfig]
        public EulerAncestralDiscreteScheduler(
            int num_train_timesteps = 1000,
            float beta_start = 0.0001f,
            float beta_end = 0.02f,
            string beta_schedule = "linear",
            Array trained_betas = null,
            string prediction_type = "epsilon"
        ) : base(num_train_timesteps)
        {
            if (trained_betas != null)
                betas = np.array(trained_betas).astype(np.float32);
            else if (beta_schedule == "linear")
                betas = np.linspace(beta_start, beta_end, num_train_timesteps);
            else if (beta_schedule == "scaled_linear")
                betas = np.power(np.linspace(beta_start, beta_end, num_train_timesteps), 2);
            else if (beta_schedule == "squaredcos_cap_v2")
                betas = BetasForAlphaBar(num_train_timesteps);
            else
                throw new NotImplementedException($"{beta_schedule} does is not implemented for {this.GetType()}");

            alphas = 1.0f - betas;
            NDArray alphas_cumprod = alphas.CumProd();// np.cumprod(alphas, axis: 0);//TODO: axis is borken
            _alphasCumulativeProducts = new List<float>(alphas_cumprod.ToArray<float>());
            NDArray sigmas = ((1 - alphas_cumprod) / alphas_cumprod).Sqrt();
            sigmas = np.concatenate(new[] { sigmas.Reverse(), np.array(0.0f) }).astype(np.float32);
            this.sigmas = sigmas;

            // standard deviation of the initial noise distribution
            InitNoiseSigma = sigmas.max();

            // setable values
            num_inference_steps = 0;
            var timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps).Reverse().ToArray<float>().Select(f => (int)f).ToList();//.Copy()
            this.Timesteps = timesteps;
            prediction_Type = prediction_type;
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

        //public int[] SetTimesteps(int num_inference_steps)
        //{
        //    double start = 0;
        //    double stop = num_train_timesteps - 1;
        //    double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

        //    this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

        //    var sigmas = alphas_cumprod.ToArray<float>().Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
        //    var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
        //    sigmas = LMSDiscreteScheduler.Interpolate(timesteps, range, sigmas).ToList();
        //    this.Sigmas = new DenseTensor<float>(sigmas.Count());
        //    for (int i = 0; i < sigmas.Count(); i++)
        //    {
        //        this.Sigmas[i] = (float)sigmas[i];
        //    }
        //    return this.Timesteps.ToArray();
        //}

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
                predOriginalSample = TensorHelper.SubtractTensors(sample.ToArray(),
                    TensorHelper.MultipleTensorByFloat(modelOutput.ToArray(),
                        sigma,
                        modelOutput.Dimensions.ToArray())
                    .ToArray(), sample.Dimensions.ToArray());// sample - sigma * modelOutput;
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

            float sigmaFrom = this.sigmas[stepIndex];
            float sigmaTo = this.sigmas[stepIndex + 1];
            float sigmaUp = MathF.Pow(sigmaTo * sigmaTo * (sigmaFrom * sigmaFrom - sigmaTo * sigmaTo) / sigmaFrom * sigmaFrom, 0.5f);
            float sigmaDown = MathF.Pow(sigmaTo * sigmaTo - sigmaUp * sigmaUp, 0.5f);

            // 2. Convert to an ODE derivative
            DenseTensor<float> derivative = TensorHelper.MultipleTensorByFloat(
                TensorHelper.SubtractTensors(sample, predOriginalSample),
                1f / sigma);// (sample - predOriginalSample) / sigma;

            float dt = sigmaDown - sigma;

            DenseTensor<float> prevSample = TensorHelper.AddTensors(sample,
                TensorHelper.MultipleTensorByFloat(derivative, dt));// sample + derivative * dt;

            //var noise = generator == null ? np.random.randn(modelOutput.shape) : np.random.RandomState(generator).randn(modelOutput.shape);
            var noise = GetRandomTensor(prevSample.Dimensions);

            prevSample = TensorHelper.AddTensors(prevSample,
                TensorHelper.MultipleTensorByFloat(noise, sigmaUp));// prevSample + noise * sigmaUp;

            //if (!returnDict)
            //{
            //    return prevSample;
            //}

            return prevSample;// new EulerAncestralDiscreteSchedulerOutput(prevSample, predOriginalSample);
        }

        public DenseTensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            float[] data = new float[dimensions.ToArray().Aggregate((d1, d2) => d1 * d2)];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Random.Shared.NextSingle();
            }
            return new DenseTensor<float>(data, dimensions);
        }
    }
}
