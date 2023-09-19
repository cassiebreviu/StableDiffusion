using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class TensorHelper
    {
        public static DenseTensor<T> CreateTensor<T>(T[] data, ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<T>(data, dimensions);
        }

        public static DenseTensor<float> DivideTensorByFloat(Tensor<float> data, float value, ReadOnlySpan<int> dimensions)
        {
            var divTensor = new DenseTensor<float>(dimensions);
            for (int i = 0; i < data.Length; i++)
            {
                divTensor.SetValue(i, data.GetValue(i) / value);
            }
            return divTensor;
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value, ReadOnlySpan<int> dimensions)
        {
            var mullTensor = new DenseTensor<float>(dimensions);
            for (int i = 0; i < data.Length; i++)
            {
                mullTensor.SetValue(i, data.GetValue(i) * value);
            }
            return mullTensor;
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
        {
            return MultipleTensorByFloat(data, value, data.Dimensions);
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor, ReadOnlySpan<int> dimensions)
        {
            var addTensor = new DenseTensor<float>(dimensions);
            for (var i = 0; i < sample.Length; i++)
            {
                addTensor.SetValue(i, sample.GetValue(i) + sumTensor.GetValue(i));
            }
            return addTensor;
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
        {
            return AddTensors(sample, sumTensor, sample.Dimensions);
        }

        public static Tuple<Tensor<float>, Tensor<float>> SplitTensor(Tensor<float> tensorToSplit, ReadOnlySpan<int> dimensions)
        {
            var tensor1 = new DenseTensor<float>(dimensions);
            var tensor2 = new DenseTensor<float>(dimensions);

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 512 / 8; k++)
                    {
                        for (int l = 0; l < 512 / 8; l++)
                        {
                            tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                            tensor2[i, j, k, l] = tensorToSplit[i, j + 4, k, l];
                        }
                    }
                }
            }
            return new Tuple<Tensor<float>, Tensor<float>>(tensor1, tensor2);

        }

        public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, ReadOnlySpan<int> dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            for (int m = 0; m < tensorArray.Length; m++)
            {
                var tensorToSum = tensorArray[m];
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumTensor.SetValue(i, sumTensor.GetValue(i) + tensorToSum.GetValue(i));
                }
            }
            return sumTensor;
        }

        public static DenseTensor<float> Duplicate(Tensor<float> data, ReadOnlySpan<int> dimensions)
        {
            var dupTensor = data.Concat(data).ToArray();
            return CreateTensor(dupTensor, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor, ReadOnlySpan<int> dimensions)
        {
            var result = new DenseTensor<float>(dimensions);
            for (var i = 0; i < sample.Length; i++)
            {
                result.SetValue(i, sample.GetValue(i) - subTensor.GetValue(i));
            }
            return result;
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
        {
            return SubtractTensors(sample, subTensor, sample.Dimensions);
        }

        public static DenseTensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            var random = new Random();
            var latents = new DenseTensor<float>(dimensions);
            for (int i = 0; i < latents.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latents.SetValue(i, (float)standardNormalRand);
            }
            return latents;
        }

    }
}
