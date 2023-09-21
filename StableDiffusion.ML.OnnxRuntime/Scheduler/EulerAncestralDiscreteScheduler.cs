using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace StableDiffusion.ML.OnnxRuntime.Scheduler
{
    public class EulerAncestralDiscreteScheduler : SchedulerBase
    {
        public EulerAncestralDiscreteScheduler() : base(new SchedulerConfig()) { }
        public EulerAncestralDiscreteScheduler(SchedulerConfig configuration) : base(configuration) { }

        public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
        {
            if (!_isScaleInputCalled)
                throw new Exception("The `scale_model_input` function should be called before `step` to ensure correct denoising. ");

            var stepIndex = _timesteps.IndexOf(timestep);
            var sigma = _sigmasTensor[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = TensorHelper.SubtractTensors(sample, TensorHelper.MultipleTensorByFloat(modelOutput, sigma));

            var sigmaFrom = _sigmasTensor[stepIndex];
            var sigmaTo = _sigmasTensor[stepIndex + 1];

            var sigmaFromLessSigmaTo = MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2);
            var sigmaUpResult = MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2);
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative
            var sampleMinusPredOriginalSample = TensorHelper.SubtractTensors(sample, predOriginalSample);
            var derivative = TensorHelper.DivideTensorByFloat(sampleMinusPredOriginalSample, sigma, predOriginalSample.Dimensions);

            var dt = sigmaDown - sigma;
            var prevSample = TensorHelper.AddTensors(sample, TensorHelper.MultipleTensorByFloat(derivative, dt));

            var noise = TensorHelper.GetRandomTensor(prevSample.Dimensions);
            var noiseSigmaUpProduct = TensorHelper.MultipleTensorByFloat(noise, sigmaUp);
            prevSample = TensorHelper.AddTensors(prevSample, noiseSigmaUpProduct);
            return prevSample;
        }
    }
}
