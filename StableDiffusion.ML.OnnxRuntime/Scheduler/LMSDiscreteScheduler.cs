using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime.Scheduler
{
    public class LMSDiscreteScheduler : SchedulerBase
    {
        private readonly List<Tensor<float>> _derivatives;

        public LMSDiscreteScheduler() : this(new SchedulerConfig()) { }

        public LMSDiscreteScheduler(SchedulerConfig configuration) : base(configuration)
        {
            _derivatives = new List<Tensor<float>>();
        }

        public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
        {
            int stepIndex = _timesteps.IndexOf(timestep);
            var sigma = _sigmasTensor[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = new DenseTensor<float>(modelOutput.Dimensions);
            for (int i = 0; i < modelOutput.Length; i++)
            {
                predOriginalSample.SetValue(i, sample.GetValue(i) - sigma * modelOutput.GetValue(i));
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(sample.Dimensions);
            for (int i = 0; i < modelOutput.Length; i++)
            {
                derivativeItems.SetValue(i, (sample.GetValue(i) - predOriginalSample.GetValue(i)) / sigma);
            }

            _derivatives.Add(derivativeItems);
            if (_derivatives.Count > order)
            {
                // remove first element
                _derivatives.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder));

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(_derivatives);

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs
                .Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative))
                .ToArray();

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[_derivatives.Count];

            for (int i = 0; i < lmsCoeffsAndDerivatives.Length; i++)
            {
                // Multiply to coeff by each derivatives to create the new tensors
                var (lmsCoeff, derivative) = lmsCoeffsAndDerivatives[i];
                lmsDerProduct[i] = TensorHelper.MultipleTensorByFloat(derivative, (float)lmsCoeff);
            }

            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            return TensorHelper.AddTensors(sample, sumTensor);
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
                    prod *= (tau - _sigmasTensor[t - k]) / (_sigmasTensor[t - currentOrder] - _sigmasTensor[t - k]);
                }
                return prod;
            }
            return Integrate.OnClosedInterval(LmsDerivative, _sigmasTensor[t], _sigmasTensor[t + 1], 1e-4);
        }
    }
}
