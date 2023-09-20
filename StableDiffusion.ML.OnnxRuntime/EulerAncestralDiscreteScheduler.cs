using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class EulerAncestralDiscreteScheduler : SchedulerBase
    {
        public EulerAncestralDiscreteScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f, string betaSchedule = "scaled_linear", string predictionType = "epsilon", List<float> trainedBetas = null)
            : base(numTrainTimesteps, betaStart, betaEnd, betaSchedule, predictionType, trainedBetas)
        {
            var alphas = new List<float>();
            var betas = new List<float>();

            if (_trained_betas != null)
            {
                betas = _trained_betas;
            }
            else if (_beta_schedule == "linear")
            {
                betas = Enumerable.Range(0, _numTrainTimesteps).Select(i => _beta_start + (_beta_end - _beta_start) * i / (_numTrainTimesteps - 1)).ToList();
            }
            else if (_beta_schedule == "scaled_linear")
            {
                var start = (float)Math.Sqrt(_beta_start);
                var end = (float)Math.Sqrt(_beta_end);
                betas = np.linspace(start, end, _numTrainTimesteps).ToArray<float>().Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            alphas = betas.Select(beta => 1 - beta).ToList();

            _alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            InitNoiseSigma = (float)sigmas.Max();
        }


        public override List<int> Timesteps { get; set; } = new List<int>();
        public override Tensor<float> Sigmas { get; set; }
        public override float InitNoiseSigma { get; set; }


        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            InitNoiseSigma = (float)sigmas.Max();
            Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                Sigmas[i] = (float)sigmas[i];
            }
            return Timesteps.ToArray();

        }


        public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
        {

            if (!is_scale_input_called)
            {
                Console.WriteLine(
                    "The `scale_model_input` function should be called before `step` to ensure correct denoising. " +
                    "See `StableDiffusionPipeline` for a usage example."
                );
            }


            int stepIndex = Timesteps.IndexOf((int)timestep);
            var sigma = Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample = null;
            if (_predictionType == "epsilon")
            {
                //  pred_original_sample = sample - sigma * model_output
                predOriginalSample = TensorHelper.SubtractTensors(sample, TensorHelper.MultipleTensorByFloat(modelOutput, sigma));
            }
            else if (_predictionType == "v_prediction")
            {
                // * c_out + input * c_skip
                //predOriginalSample = modelOutput * (-sigma / Math.Pow(sigma * sigma + 1, 0.5)) + (sample / (sigma * sigma + 1));
                throw new NotImplementedException($"prediction_type not implemented yet: {_predictionType}");
            }
            else if (_predictionType == "sample")
            {
                throw new NotImplementedException($"prediction_type not implemented yet: {_predictionType}");
            }
            else
            {
                throw new ArgumentException($"prediction_type given as {_predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            float sigmaFrom = Sigmas[stepIndex];
            float sigmaTo = Sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = (MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2));
            var sigmaUpResult = (MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo) / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = (MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2));
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative
            var sampleMinusPredOriginalSample = TensorHelper.SubtractTensors(sample, predOriginalSample);
            DenseTensor<float> derivative = TensorHelper.DivideTensorByFloat(sampleMinusPredOriginalSample, sigma, predOriginalSample.Dimensions);// (sample - predOriginalSample) / sigma;

            float dt = sigmaDown - sigma;

            DenseTensor<float> prevSample = TensorHelper.AddTensors(sample, TensorHelper.MultipleTensorByFloat(derivative, dt));// sample + derivative * dt;

            //var noise = generator == null ? np.random.randn(modelOutput.shape) : np.random.RandomState(generator).randn(modelOutput.shape);
            var noise = TensorHelper.GetRandomTensor(prevSample.Dimensions);

            var noiseSigmaUpProduct = TensorHelper.MultipleTensorByFloat(noise, sigmaUp);
            prevSample = TensorHelper.AddTensors(prevSample, noiseSigmaUpProduct);// prevSample + noise * sigmaUp;

            return prevSample;
        }

    }
}
