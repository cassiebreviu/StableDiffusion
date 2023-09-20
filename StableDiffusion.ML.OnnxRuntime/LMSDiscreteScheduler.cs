using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class LMSDiscreteScheduler : SchedulerBase
    {
        public LMSDiscreteScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f, string betaSchedule = "scaled_linear", string predictionType = "epsilon", List<float> trainedBetas = null)
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

        public List<Tensor<float>> Derivatives { get; set; } = new List<Tensor<float>>();
        public override List<int> Timesteps { get; set; } = new List<int>();
        public override Tensor<float> Sigmas { get; set; }
        public override float InitNoiseSigma { get; set; }


        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                Sigmas[i] = (float)sigmas[i];
            }
            return Timesteps.ToArray();
        }


        public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
        {
            int stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample;

            // Create array of type float length modelOutput.length
            float[] predOriginalSampleArray = new float[modelOutput.Length];
            var modelOutPutArray = modelOutput.ToArray();
            var sampleArray = sample.ToArray();

            if (_predictionType == "epsilon")
            {

                for (int i = 0; i < modelOutPutArray.Length; i++)
                {
                    predOriginalSampleArray[i] = sampleArray[i] - sigma * modelOutPutArray[i];
                }
                predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, modelOutput.Dimensions);

            }
            else if (_predictionType == "v_prediction")
            {
                //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                throw new Exception($"prediction_type given as {_predictionType} not implemented yet.");
            }
            else
            {
                throw new Exception($"prediction_type given as {_predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(sample.Dimensions);

            var derivativeItemsArray = new float[derivativeItems.Length];

            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItemsArray[i] = (sampleArray[i] - predOriginalSampleArray[i]) / sigma;
            }
            derivativeItems = TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions);

            Derivatives?.Add(derivativeItems);

            if (Derivatives?.Count() > order)
            {
                // remove first element
                Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors derivatives
            var revDerivatives = Enumerable.Reverse(Derivatives).ToList();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[Derivatives.Count()];

            for (int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
            {
                var item = lmsCoeffsAndDerivatives.ElementAt(m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(item.derivative, (float)item.lmsCoeff, item.derivative.Dimensions);
            }
            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            var prevSample = TensorHelper.AddTensors(sample, sumTensor, sample.Dimensions);

            Console.WriteLine(prevSample[0]);
            return prevSample;
        }


        //python line 135 of scheduling_lms_discrete.py
        private double GetLmsCoefficient(int order, int t, int currentOrder)
        {
            // Compute a linear multistep coefficient.

            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - Sigmas[t - k]) / (Sigmas[t - currentOrder] - Sigmas[t - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, Sigmas[t], Sigmas[t + 1], 1e-4);

            return integratedCoeff;
        }
    }
}
