using Microsoft.ML.OnnxRuntime.Tensors;
using MathNet.Numerics;
using NumSharp;
using System.Runtime.Serialization.Formatters;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class LMSDiscreteScheduler : SchedulerBase
    {
        private int _numTrainTimesteps;

        private string _predictionType;

        public override Tensor<Float16> Sigmas { get; set; }
        public override List<Float16> Timesteps { get; set; }
        public List<Tensor<Float16>> Derivatives;
        public override float InitNoiseSigma { get; set; }

        public LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, string beta_schedule = "scaled_linear", string prediction_type = "epsilon", List<Float16> trained_betas = null)
        {
            _numTrainTimesteps = num_train_timesteps;
            _predictionType = prediction_type;
            Derivatives = new List<Tensor<Float16>>();
            Timesteps = new List<Float16>();

            var alphas = new List<Float16>();
            var betas = new List<Float16>();

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                for(int i = 0; i < num_train_timesteps; i++)
                {
                    betas.Add((Float16)(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)));
                }
                //betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start =(Float16)Math.Sqrt(beta_start);
                var end =(Float16)Math.Sqrt(beta_end);
                betas = np.linspace(start, end, num_train_timesteps).ToArray<Float16>().ToList();
                for(int i = 0; i < betas.Count(); i++)
                {
                    betas[i] = (Float16)(betas[i] * betas[i]);
                }
                //.Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            for(int i = 0; i < betas.Count(); i++)
            {
                alphas.Add((Float16)(1 - betas[i]));
                //alphas = betas.Select(beta => 1 - beta).ToList();
            }
            
 
            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate<Float16>((a, b) => (Float16)(a * b))).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma =(Float16)sigmas.Max();

        }

        //python line 135 of scheduling_lms_discrete.py
        public double GetLmsCoefficient(int order, int t, int currentOrder)
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
                    prod *= (tau - this.Sigmas[t - k]) / (this.Sigmas[t - currentOrder] - this.Sigmas[t - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[t], this.Sigmas[t + 1], 1e-4);

            return integratedCoeff;
        }

        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override Float16[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (Float16)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.Sigmas = new DenseTensor<Float16>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] =(Float16)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public override DenseTensor<Float16> Step(
               Tensor<Float16> modelOutput,
               Float16 timestep,
               Tensor<Float16> sample,
               int order = 4)
        {
            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<Float16> predOriginalSample;

            // Create array of type float length modelOutput.length
            Float16[] predOriginalSampleArray = new Float16[modelOutput.Length];
            var modelOutPutArray = modelOutput.ToArray();
            var sampleArray = sample.ToArray();

            if (this._predictionType == "epsilon")
            {

                for (int i=0; i < modelOutPutArray.Length; i++)
                {
                    predOriginalSampleArray[i] = (Float16)(sampleArray[i] - sigma * modelOutPutArray[i]);
                }
                predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, modelOutput.Dimensions.ToArray());

            }
            else if (this._predictionType == "v_prediction")
            {
                //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                throw new Exception($"prediction_type given as {this._predictionType} not implemented yet.");
            }
            else
            {
                throw new Exception($"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<Float16>(sample.Dimensions.ToArray());

            var derivativeItemsArray = new Float16[derivativeItems.Length];
            
            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItemsArray[i] = (Float16)((sampleArray[i] - predOriginalSampleArray[i]) / sigma);
            }
            derivativeItems =  TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

            this.Derivatives?.Add(derivativeItems);

            if (this.Derivatives?.Count() > order)
            {
                // remove first element
                this.Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(this.Derivatives).ToList();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<Float16>[this.Derivatives.Count()];

            for(int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
            {
                var item = lmsCoeffsAndDerivatives.ElementAt(m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(item.derivative.ToArray(),(Float16)item.lmsCoeff, item.derivative.Dimensions.ToArray());
            }
            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            var prevSample = TensorHelper.AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());

            Console.WriteLine(prevSample[0]);
            return prevSample;

        }
    }
}
