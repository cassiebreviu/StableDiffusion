﻿using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion
{
    public class TensorHelper
    {
        public static DenseTensor<T> CreateTensor<T>(T[] data, int[] dimensions)
        {
            return new DenseTensor<T>(data, dimensions); ;
        }
        
        public static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] / value;
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] * value;
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
        {
            return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
        }

        public static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions)
        {
            for(var i=0; i < sample.Length; i++)
            {
                sample[i] = sample[i] + sumTensor[i];
            }
            return CreateTensor(sample, dimensions); ;
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
        {
            return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static Tuple<Tensor<float>, Tensor<float>> SplitTensor(Tensor<float> tensorToSplit, int[] dimensions)
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

        public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            var sumArray = new float[sumTensor.Length];

            for (int m = 0; m < tensorArray.Count(); m++)
            {
                var tensorToSum = tensorArray[m].ToArray();
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumArray[i] += (float)tensorToSum[i];
                }
            }

            return CreateTensor(sumArray, dimensions);
        }

        public static DenseTensor<float> Duplicate(float[] data, int[] dimensions)
        {
            data = data.Concat(data).ToArray();
            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(float[] sample, float[] subTensor, int[] dimensions)
        {
            for (var i = 0; i < sample.Length; i++)
            {
                sample[i] = sample[i] - subTensor[i];
            }
            return CreateTensor(sample, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
        {
            return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
        }
    }
}
