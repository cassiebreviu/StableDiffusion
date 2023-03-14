using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public static class SafetyChecker
    {
        public static int IsSafe(Tensor<float> resultImage)
        {

            var safetyModelPath = Directory.GetCurrentDirectory().ToString() + ("\\safety_checker\\model.onnx");
            var safetySession = new InferenceSession(safetyModelPath);

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("clip_input", resultImage)};
            
            // Run session and send the input data in to get inference output. 
            var output = safetySession.Run(input);
            var result = (output.ToList().First().Value as IEnumerable<int>).ToArray()[0];

            return result;
        }
    }
}
