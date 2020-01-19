using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Mathematics;

namespace NeuralBurst.Dev
{
    public class MathTesting
    {


        public static float[] Softmax(float[] testData)
        {
            float[] result = new float[testData.Length];

            float sum = 0.0f;
            for (int i = 0; i < testData.Length; i++)
            {
                sum += math.exp(testData[i]);
            }

            for (int i = 0; i < testData.Length; i++)
            {
                result[i] = math.exp(testData[i]) / sum;
            }

            return result;
        }

    }
}
