using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Mathematics;

namespace NeuralBurst
{
    public static class MathOperations
    {

        public static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + math.exp(-x));
        }

        public static float SigmoidPrime(float x)
        {
            return Sigmoid(x) * (1.0f - Sigmoid(x));
        }

        public static float QuadraticCost(NativeArray<float> expected, NativeArray<float> actual)
        {
            float sum = 0.0f;
            for (int i = 0; i < expected.Length; i++)
            {
                sum += expected[i] - actual[i];
            }

            return 0.5f * sum;
        }
    }
}
