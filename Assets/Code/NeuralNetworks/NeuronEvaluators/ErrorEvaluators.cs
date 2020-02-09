using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace NeuralBurst
{
    public class ErrorEvaluators
    {


/*        [BurstCompile] //TODO:Probably needs to be split, so that the cost function and the layer derivitive from activation are computed seperately, and then multiplied
                       //However it's going to be faster to manually merge the jobs at some point
        public struct QuadraticSigmoidOutputErrorEvaluator : IJobParallelFor
        {
            [ReadOnly]
            public TestDataSlice Expected;
            [ReadOnly]
            public NativeArray<float> Actuall;
            [ReadOnly]
            public NativeArray<float> WeightedActivation;

            public NativeArray<float> ErrorOut;

            public void Execute(int index)
            {
                float deltaC = Actuall[index] - Expected[0, index];
                float sigmoidPrime = MathOperations.SigmoidPrime(WeightedActivation[index]);
                ErrorOut[index] = deltaC * sigmoidPrime;
            }
        }*/

        [BurstCompile] //TODO:Probably needs to be split, so that the cost function and the layer derivitive from activation are computed seperately, and then multiplied
        //However it's going to be faster to manually merge the jobs at some point
        public struct CrossEntropySigmoidOutputErrorEvaluator : IJobParallelFor
        {
            [ReadOnly]
            public NativeSlice2D<float> Expected;
            [ReadOnly]
            public NativeSlice2D<float> Actuall;
            [ReadOnly]
            public NativeSlice2D<float> WeightedActivation;

            public NativeSlice2D<float> ErrorOut;

            public void Execute(int index)
            {
                for (int view = 0; view < Actuall.Dimensions.x; view++)
                {
                    float deltaC = Actuall[view,index] - Expected[view, index];
                    ErrorOut[view, index] = deltaC;
                }
            }
        }

        [BurstCompile]
        public struct AccumulateGradientOverWeight : IJobParallelFor
        {
            [ReadOnly]
            public NativeSlice2D<float> PreviousActivation;
            [ReadOnly]
            public NativeSlice2D<float> NextError;

            //ForEach over this
            public NativeArray<float> WeightGradients;//For next

            public void Execute(int index)
            {
                int nextIndex = index / PreviousActivation.Dimensions.y;// for size 50 in next error, first 50 weights point from everything in previous activation to this node
                int previousIndex = index % PreviousActivation.Dimensions.y;

                //int srcIndex = index % NextError.Length;// for size 50 in next error, first 50 weights point from everything in previous activation to this node
                //int dstIndex = index / NextError.Length;

                //float weightGradient = WeightGradients[index];//TODO:Use this
                float weightGradient = 0.0f;
                for (int view = 0; view < PreviousActivation.Dimensions.x; view++)
                {
                    var previousActivation = PreviousActivation[view, previousIndex];
                    var nextError = NextError[view, nextIndex];

                    if (math.isnan(previousActivation))
                    {
                        throw new InvalidOperationException($"previous activation is nan at {view} {previousIndex}");
                    }


                    if (math.isnan(nextError))
                    {
                        throw new InvalidOperationException($"next error is nan at {view} {nextIndex}");
                    }

                    weightGradient += previousActivation * nextError;
                }

                WeightGradients[index] = weightGradient;
            }
        }

        [BurstCompile]
        public struct ApplyGradientToLayerWeights : IJobParallelFor
        {
            public int TestCount;
            public float LearningRate;

            [ReadOnly]
            public NativeArray<float> WeightGradients;

            public NativeArray<float> LayerWeights;

            public void Execute(int index)
            {
                var currentGradient = WeightGradients[index];

                if (math.isnan(currentGradient))
                {
                    throw new InvalidOperationException($"Gradient is nan at {index}");
                }

                LayerWeights[index] = LayerWeights[index] - LearningRate * (currentGradient / TestCount);
            }
        }

        [BurstCompile]
        public struct ApplyGradientToLayerBiases : IJobParallelFor
        {
            public int TestCount;
            public float LearningRate;

            [ReadOnly]
            public NativeSlice2D<float> LayerErrors;

            public NativeArray<float> LayerBiases;

            public void Execute(int index)
            {
                var bias = LayerBiases[index];
                for(int view = 0; view < LayerErrors.Dimensions.x; view++)
                {
                    bias = bias - LearningRate * (LayerErrors[view, index]);
                }
                LayerBiases[index] = bias;
            }
        }

        /*        [BurstCompile] //TODO:Probably needs to be split, so that the cost function and the layer derivitive from activation are computed seperately, and then multiplied
                //However it's going to be faster to manually merge the jobs at some point
                public struct CrossEntropySigmoidOutputErrorEvaluator : IJobParallelFor
                {
                    [ReadOnly]
                    public TestDataSlice Expected;
                    [ReadOnly]
                    public NativeArray<float> Actuall;
                    [ReadOnly]
                    public NativeArray<float> WeightedActivation;

                    public NativeArray<float> ErrorOut;

                    public void Execute(int index)
                    {
                        float deltaC = Actuall[index] - Expected[0, index];
                        ErrorOut[index] = deltaC;
                    }
                }

                [BurstCompile]
                public struct AccumulateGradientOverWeight : IJobParallelFor
                {
                    [ReadOnly]
                    public NativeArray<float> PreviousActivation;
                    [ReadOnly]
                    public NativeArray<float> NextError;

                    //ForEach over this
                    public NativeArray<float> WeightGradients;//For next


                    public void Execute(int index)
                    {
                        int nextIndex = index / PreviousActivation.Length;// for size 50 in next error, first 50 weights point from everything in previous activation to this node
                        int previousIndex = index % PreviousActivation.Length;

                        //int srcIndex = index % NextError.Length;// for size 50 in next error, first 50 weights point from everything in previous activation to this node
                        //int dstIndex = index / NextError.Length;

                        //TODO:Accumulate
                        WeightGradients[index] = PreviousActivation[previousIndex] * NextError[nextIndex];
                    }
                }

                [BurstCompile]
                public struct ApplyGradientToLayerWeights : IJobParallelFor
                {
                    public int TestCount;
                    public float LearningRate;

                    [ReadOnly]
                    public NativeArray<float> WeightGradients;

                    public NativeArray<float> LayerWeights;

                    public void Execute(int index)
                    {
                        LayerWeights[index] = LayerWeights[index] - LearningRate * (WeightGradients[index]);
                    }
                }

                [BurstCompile]
                public struct ApplyGradientToLayerBiases : IJobParallelFor
                {
                    public int TestCount;
                    public float LearningRate;

                    [ReadOnly]
                    public NativeArray<float> LayerErrors;

                    public NativeArray<float> LayerBiases;


                    public void Execute(int index)
                    {
                        LayerBiases[index] = LayerBiases[index] -  LearningRate * (LayerErrors[index]);
                    }
                }*/
    }
}
