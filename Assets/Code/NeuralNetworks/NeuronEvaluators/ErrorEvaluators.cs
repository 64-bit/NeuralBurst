using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace NeuralBurst
{
    public class ErrorEvaluators
    {


        [BurstCompile]
        public struct QuadraticSigmoidOutputErrorEvaluator : IJobParallelFor
        {
            public NativeArray<float> Expected;
            public NativeArray<float> Actuall;
            public NativeArray<float> WeightedActivation;

            public NativeArray<float> ErrorOut;

            public void Execute(int index)
            {
                float deltaC = Actuall[index] - Expected[index];
                float sigmoidPrime = MathOperations.SigmoidPrime(WeightedActivation[index]);
                ErrorOut[index] = deltaC * sigmoidPrime;
            }
        }

        [BurstCompile]
        public struct SigmoidLayerErrorEvaluator : IJobParallelFor
        {
            public NativeArray<float> NextLayerWeights;
            public NativeArray<float> NextLayerError;
            public NativeArray<float> WeightedActivation;

            public NativeArray<float> ErrorOutput;

            public void Execute(int index)
            {
                //Multiply the error of every single next node by the weight connecting us to them
                int nextLayerSize = NextLayerError.Length;

                float backpropigatedError = 0.0f;
                for (int i = 0; i < nextLayerSize; i++)
                {
                    float nextNodeError = NextLayerError[i];
                    float weightToNextNode = NextLayerWeights[index + i * nextLayerSize];

                    backpropigatedError += nextNodeError * weightToNextNode;
                }

                //Multiply this by the rate of change of our own activation function
                float deltaActivation = MathOperations.SigmoidPrime(WeightedActivation[index]);

                //TODO:May need to accumulate this error
                ErrorOutput[index] = backpropigatedError * deltaActivation;
            }
        }

        [BurstCompile]
        public struct AccumulateGradientOverWeight : IJobParallelFor
        {
            public NativeArray<float> PreviousActivation;
            public NativeArray<float> NextError;

            //ForEach over this
            public NativeArray<float> WeightGradients;//For next


            public void Execute(int index)
            {
                int srcIndex = index / NextError.Length;// for size 50 in next error, first 50 weights point from everything in previous activation to this node
                int dstIndex = index % NextError.Length;

                //TODO:Accumulate
                WeightGradients[index] = PreviousActivation[srcIndex] * NextError[dstIndex];
            }
        }

        [BurstCompile]
        public struct ApplyGradientToLayerWeights : IJobParallelFor
        {
            public int TestCount;
            public float LearningRate;

            public NativeArray<float> LayerWeights;

            public NativeArray<float> WeightGradients;

            public void Execute(int index)
            {
                LayerWeights[index] -= LearningRate * (WeightGradients[index] / TestCount);
            }
        }

        [BurstCompile]
        public struct ApplyGradientToLayerBiases : IJobParallelFor
        {
            public int TestCount;
            public float LearningRate;

            public NativeArray<float> LayerBiases;
            public NativeArray<float> LayerErrors;

            public void Execute(int index)
            {
                LayerBiases[index] -= LearningRate * (LayerErrors[index] / TestCount);
            }
        }
    }
}
