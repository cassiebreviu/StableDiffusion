﻿using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class TextProcessing
    {
        public static DenseTensor<float> PreprocessText(String prompt, StableDiffusionConfig config)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt, config);
            var textPromptEmbeddings = TextEncoder(textTokenized, config).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = CreateUncondInput();
            var uncondEmbedding = TextEncoder(uncondInputTokens, config).ToArray();

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            }
            return textEmbeddings;
        }
        public static int[] TokenizeText(string text, StableDiffusionConfig config)
        {
            // Create session options for custom op of extensions
            using (var sessionOptions = new SessionOptions())
            {
                sessionOptions.RegisterOrtExtensions();

                // Create an InferenceSession from the onnx clip tokenizer.
                using (var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions))
                {
                    var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
                    var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

                    // Run session and send the input data in to get inference output. 
                    using (var tokens = tokenizeSession.Run(inputString))
                    {
                        var inputIds = tokens.FirstElementAs<IEnumerable<long>>();
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
                }
            }
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

            // Run inference.
            using (var sessionOptions = config.GetSessionOptionsForEp())
            using (var encodeSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions))
            using (var encoded = encodeSession.Run(input))
            {
                var lastHiddenState = encoded.FirstElementAs<IEnumerable<float>>();
                var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

                return lastHiddenStateTensor;
            }
        }

    }
}