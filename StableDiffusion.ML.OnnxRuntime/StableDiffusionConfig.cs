using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class StableDiffusionConfig
    {
        public enum ExecutionProvider
        {
            DirectML = 0,
            Cuda = 1,
            Cpu = 2,
            NNPI = 3,
            CoreML = 4,
            SNPE = 5
        }

        // default props
        public int NumInferenceSteps = 15;
        public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.Cuda;
        public double GuidanceScale = 7.5;
        public int Height = 512;
        public int Width = 512;
        public int DeviceId = 0;


        public string TokenizerOnnxPath = "cliptokenizer.onnx";
        public string TextEncoderOnnxPath = "";
        public string UnetOnnxPath = "";
        public string VaeDecoderOnnxPath = "";
        public string SafetyModelPath = "";

        // default directory for images
        public string ImageOutputPath = "";

        public SessionOptions GetSessionOptionsForEp()
        {
            var sessionOptions = new SessionOptions();


            switch (this.ExecutionProviderTarget)
            {
                case ExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.NNPI:
                    sessionOptions.AppendExecutionProvider_Nnapi(
                        NnapiFlags.NNAPI_FLAG_USE_FP16 |
                        NnapiFlags.NNAPI_FLAG_CPU_DISABLED);
                    return sessionOptions;
                case ExecutionProvider.CoreML:
                    sessionOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE
                    );
                    return sessionOptions;
                case ExecutionProvider.SNPE:
                    sessionOptions.AppendExecutionProvider("SNPE", new Dictionary<string, string>()
                    {
                        { "runtime", "DSP" },
                        { "buffer_type", "FLOAT" }
                    });
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    //default to CUDA, fall back on CPU if CUDA is not available.
                    sessionOptions.AppendExecutionProvider_CUDA(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                    return sessionOptions;
            }
        }
    }
}