using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public static class SafetyChecker
    {
        public static int IsSafe(Tensor<float> resultImage, StableDiffusionConfig config)
        {

            var sessionOptions = config.GetSessionOptionsForEp();
            var safetySession = new InferenceSession(config.SafetyModelPath, sessionOptions);

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("clip_input", resultImage)};
            
            // Run session and send the input data in to get inference output. 
            var output = safetySession.Run(input);
            var result = (output.ToList().First().Value as IEnumerable<int>).ToArray()[0];

            return result;
        }
    }
}
