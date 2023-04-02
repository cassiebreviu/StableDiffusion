using NumSharp;

namespace StableDiffusion
{
    public static class NumSharpExtensions
    {
        public static NDArray CumProd(this NDArray a, int axis = 0)
        {
            float[] values = a.ToArray<float>();
            float[] result = new float[values.Length];
            result[0] = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                result[i] = values[i] * values[i - 1];
            }

            return new NDArray(result, a.shape);
        }

        public static NDArray Sqrt(this NDArray a)
        {
            float[] values = a.ToArray<float>();
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Sqrt(values[i]);
            }
            return new NDArray(values, a.shape);
        }

        public static NDArray Reverse(this NDArray a)
        {
            float[] values = a.ToArray<float>();
            Array.Reverse(values);
            return new NDArray(values, a.shape);
        }
    }
}
