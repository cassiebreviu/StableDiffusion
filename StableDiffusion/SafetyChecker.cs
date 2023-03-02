using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public static class SafetyChecker
    {
        public static int IsSafe(Tensor<float> resultImage)
        {
            // Load the autoencoder model which will be used to decode the latents into image space. 
            var safetyModelPath = @"C:\code\StableDiffusion\StableDiffusion\safety_checker\model.onnx";
            var cudaProviderOptions = new OrtCUDAProviderOptions();
            // use gpu
            var providerOptionsDict = new Dictionary<string, string>();
            providerOptionsDict["device_id"] = "0";
            //providerOptionsDict["gpu_mem_limit"] = "2147483648";
            providerOptionsDict["arena_extend_strategy"] = "kSameAsRequested";
            providerOptionsDict["cudnn_conv_algo_search"] = "DEFAULT";
            providerOptionsDict["do_copy_in_default_stream"] = "1";
            providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";
            providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";

            cudaProviderOptions.UpdateOptions(providerOptionsDict);

            SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
            var safetySession = new InferenceSession(safetyModelPath, options);

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("clip_input", resultImage)};
            
            // Run session and send the input data in to get inference output. 
            var output = safetySession.Run(input);
            var result = (output.ToList().First().Value as IEnumerable<int>).ToArray()[0];

            return result;
        }
    }
}
