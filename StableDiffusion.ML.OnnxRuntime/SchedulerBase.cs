using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime
{
    public abstract class SchedulerBase
    {
        protected readonly int _numTrainTimesteps;
        protected readonly string _predictionType;
        protected readonly float _beta_start;
        protected readonly float _beta_end;
        protected readonly string _beta_schedule;
        protected readonly List<float> _trained_betas;

        protected bool is_scale_input_called;
        protected List<float> _alphasCumulativeProducts;

        public SchedulerBase(int numTrainTimesteps, float betaStart, float betaEnd, string betaSchedule, string predictionType, List<float> trainedBetas)
        {
            _numTrainTimesteps = numTrainTimesteps;
            _predictionType = predictionType;
            _beta_start = betaStart;
            _beta_end = betaEnd;
            _beta_schedule = betaSchedule;
            _trained_betas = trainedBetas;
        }

        public abstract List<int> Timesteps { get; set; }
        public abstract Tensor<float> Sigmas { get; set; }
        public abstract float InitNoiseSigma { get; set; }
        public abstract int[] SetTimesteps(int num_inference_steps);
        public abstract DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4);

        public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            // Get step index of timestep from TimeSteps
            int stepIndex = this.Timesteps.IndexOf(timestep);
            // Get sigma at stepIndex
            var sigma = this.Sigmas[stepIndex];
            sigma = (float)Math.Sqrt((Math.Pow(sigma, 2) + 1));

            // Divide sample tensor shape {2,4,64,64} by sigma
            sample = TensorHelper.DivideTensorByFloat(sample, sigma, sample.Dimensions);
            is_scale_input_called = true;
            return sample;
        }

        protected double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
        {

            // Create an output array with the same shape as timesteps
            var result = np.zeros(timesteps.Length + 1);

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[0];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[sigmas.Count - 1];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
                }

            }
            //  add 0.000 to the end of the result
            result = np.add(result, 0.000f);

            return result.ToArray<double>();
        }
    }
}