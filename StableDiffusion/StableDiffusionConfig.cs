using Microsoft.ML.OnnxRuntime;
using System.Runtime.CompilerServices;

namespace StableDiffusion
{
    public class StableDiffusionConfig
    {
        public enum ExecutionProvider
        {
            DirectML = 0,
            Cuda = 1,
            Cpu = 2,
            //OpenVINO = 4,
        }
        // default props
        public int NumInferenceSteps = 15;
        public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.Cuda;
        public double GuidanceScale = 7.5;
        public int BatchSize = 1;
        public int Height = 512;
        public int Width = 512;
        public int DeviceId = 0;


        public string OrtExtensionsPath = "";
        public string TokenizerOnnxPath = "";
        public string TextEncoderOnnxPath = "";
        public string UnetOnnxPath = "";
        public string VaeDecoderOnnxPath = "";
        public string SafetyModelPath = "";
        

        public void SetModelPaths(bool useStaticPath = false)
        {
            // For some editors the dynamic path doesnt work. If you need to change to a static path
            // update the useStaticPath to true and update the paths below.

            if (!useStaticPath)
            {
                Directory.SetCurrentDirectory(@"..\..\..\..");
                OrtExtensionsPath = Directory.GetCurrentDirectory().ToString() + ("\\ortextensions.dll");
                TokenizerOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_tokenizer\\custom_op_cliptok.onnx");
                TextEncoderOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_encoder\\model.onnx");
                UnetOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\unet\\model.onnx");
                VaeDecoderOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\vae_decoder\\model.onnx");
                SafetyModelPath = Directory.GetCurrentDirectory().ToString() + ("\\safety_checker\\model.onnx");
            }
            else
            {
                OrtExtensionsPath = "ortextensions.dll";
                TokenizerOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\text_tokenizer\custom_op_cliptok.onnx";
                TextEncoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\text_encoder\model.onnx";
                UnetOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\unet\model.onnx";
                VaeDecoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\vae_decoder\model.onnx";
                SafetyModelPath = @"C:\code\StableDiffusion\StableDiffusion\safety_checker\model.onnx";
            }

        }


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
