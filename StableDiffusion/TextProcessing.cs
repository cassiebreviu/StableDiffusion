using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion
{
    public static class TextProcessing
    {
        public static int[] TokenizeText(string text, StableDiffusionConfig config)
        {
            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterCustomOpLibraryV2(config.OrtExtensionsPath, out var libraryHandle);
            
            // Create an InferenceSession from the onnx clip tokenizer.
            var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);
            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };
            // Run session and send the input data in to get inference output. 
            var tokens = tokenizeSession.Run(inputString);


            var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
            Console.WriteLine(String.Join(" ", inputIds));

            // Cast inputIds to Int32
            var InputIdsInt = inputIds.Select(x => (int)x).ToArray();

            var modelMaxLength = 77;
            // Pad array with 49407 until length is modelMaxLength
            if (InputIdsInt.Length < modelMaxLength)
            {
                var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
                InputIdsInt = InputIdsInt.Concat(pad).ToArray();
            }

            return InputIdsInt;

        }

        public static int[] CreateUncondInput()
        {
            // Create an array of empty tokens for the unconditional input.
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            var inputIds = new List<Int32>();
            inputIds.Add(49406);
            var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            return inputIds.ToArray();
        }

        public static DenseTensor<float> TextEncoder(int[] tokenizedInput, StableDiffusionConfig config)
        {
            // Create input tensor.
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

            // Set CUDA EP
            var sessionOptions = config.GetSessionOptionsForEp();

            var encodeSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions);
            // Run inference.
            var encoded = encodeSession.Run(input);

            var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            return lastHiddenStateTensor;

        }

    }
}