using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Assets.Code.NeuralNetworks.Evaluator;
using NeuralNetworks.Utilities;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Profiling;

namespace NeuralBurst
{
    /// <summary>
    /// Used to evaluate a network
    /// </summary>
    public class NetworkEvaluator
    {
        public float LearningRate = 0.00015f;
        private int _simultaniousLayers;

        private readonly List<EvaluatorLayer> _layers = new List<EvaluatorLayer>();

        private EvaluatorInputLayer InputLayer => _layers[0] as EvaluatorInputLayer;
        private EvaluatorLayer OutputLayer => _layers[_layers.Count - 1];
        private JobHandle _lastDependentJobHandle;

        public NetworkEvaluator(NeuralNetwork network,float learningRate = 0.0015f, int simultaniousLayers = 16)
        {
            LearningRate = learningRate;
            _simultaniousLayers = simultaniousLayers;

            foreach (var layer in network.Layers)
            {
                _layers.Add(EvaluatorLayer.CreateFromLayer(layer, simultaniousLayers));
            }
        }

        public void Dispose()
        {
            foreach (var layer in _layers)
            {
                layer.Dispose();
            }
        }

        public JobHandle Evaluate(NativeSlice2D<float> inputData, NativeSlice2D<float> outputArray, int count)
        {
            Profiler.BeginSample("NetworkEvaluator::Evaluate");
            var inputLayer = _layers[0];
            var outputLayer = _layers[_layers.Count - 1];

            if (inputData.Dimensions.y != inputLayer.Size)
            {
                throw new ArgumentException();//TODO:
            }

            if (outputArray.Dimensions.y != outputLayer.Size)
            {
                throw new ArgumentException();//TODO:
            }

            var result =  EvaluateInternal(inputData, outputArray, count);
            JobHandle.ScheduleBatchedJobs();
            Profiler.EndSample();
            return result;
        }

        private JobHandle EvaluateInternal(NativeSlice2D<float> inputData, NativeSlice2D<float> outputArray, int count) //Evaluate a single instance
        {
            //Copy to input layer TODO:Just cast to input layer and set the slice on it
            //TODO:Validate size ?
            //inputData.Data.CopyTo(InputLayer.OutputActivation.Data);
            InputLayer.SetInput(inputData);

            //inputData.Data.CopyTo(InputLayer.OutputActivation); //TODO:Re-Work this to take slices

            //InputLayer.OutputActivation.CopyFrom(inputData);
            //var jobHandle = inputData.CopyToJob(InputLayer.OutputActivation, _lastDependentJobHandle);
            JobHandle jobHandle = default;
           
            for (int i = 1; i < _layers.Count; i++)
            {
                //Evaluate layers
                jobHandle = EvaluateLayer(i, jobHandle, count);
            }

            _lastDependentJobHandle = jobHandle;

            var copyToOutputJob = new CopyToOutputJob()
            {
                Source = OutputLayer.OutputActivation.Slice(0, count),
                Destination = outputArray
            };

            return copyToOutputJob.Schedule(copyToOutputJob.Source.Dimensions.x, 64, jobHandle);

            //Copy to output TODO: Before evaluating, set the output array on the output layer
            //return OutputLayer.OutputActivation.BackingStore.CopyToJob(outputArray.BackingSlice, jobHandle);
        }

        [BurstCompile]
        private struct CopyToOutputJob : IJobParallelFor
        {
            public NativeSlice2D<float> Source;
            public NativeSlice2D<float> Destination;

            public void Execute(int index)
            {
                for (int y = 0; y < Source.Dimensions.y; y++)
                {
                    Destination[index, y] = Source[index, y];
                }
            }
        }

        private JobHandle EvaluateLayer(int layerIndex, JobHandle lastJobHandle, int instanceCount)
        {
            var currentLayer = _layers[layerIndex];
            var lastLayer = _layers[layerIndex - 1];

           return currentLayer.ModelLayer.EvaluateLayer(lastLayer, currentLayer, instanceCount, lastJobHandle);
        }

        //Oh Boy
        public JobHandle GradientDescentBackpropigate(NativeSlice2D<float> inputData, NativeSlice2D<float> expectedOutput, int testCaseCount, out float errorSum)
        {
            Profiler.BeginSample("NetworkEvaluator::GradientDescentBackpropigate");
            var resultArray = new NativeArray2D<float>(testCaseCount, OutputLayer.Size);

            var feedforwardHandle = Evaluate(inputData, resultArray.Slice(0, testCaseCount), testCaseCount);
            //Now we need to use all the jobs created before to evaluate this.

            //Compute output error

/*            var computeOutputErrorJob = new ErrorEvaluators.QuadraticSigmoidOutputErrorEvaluator()
            {
                Expected = expectedOutput,
                Actuall = resultArray,
                WeightedActivation = OutputLayer.WeightedInput,
                ErrorOut = OutputLayer.Error
            };*/

            var computeOutputErrorJob = new ErrorEvaluators.CrossEntropySigmoidOutputErrorEvaluator()
            {
                Expected = expectedOutput,
                Actuall = resultArray.Slice(0, testCaseCount),
                WeightedActivation = OutputLayer.SliceWeightedInputs(0, testCaseCount),
                ErrorOut = OutputLayer.SliceError(0, testCaseCount)
            };

           
            var outputErrorHandle = computeOutputErrorJob.Schedule(OutputLayer.Error.Dimensions.y, 4, feedforwardHandle);

            //Convert that output error to the output node gradient

            //outputErrorHandle.Complete();
            //Perform backpropigation
            JobHandle backpropigationHandle = outputErrorHandle;

            for (int layerIndex = _layers.Count - 2; layerIndex >= 0; layerIndex--)
            {
                var targetLayer = _layers[layerIndex];
                var nextLayer = _layers[layerIndex + 1];
                
                if(layerIndex != 0)
                {
                    backpropigationHandle = targetLayer.ModelLayer.BackpropigateLayer(targetLayer, nextLayer, testCaseCount, backpropigationHandle);
                }

                var accumulateGradientOverWeightJob = new ErrorEvaluators.AccumulateGradientOverWeight()
                {
                    PreviousActivation = targetLayer.SliceActivations(0, testCaseCount),
                    NextError = nextLayer.SliceError(0, testCaseCount),

                    WeightGradients = nextLayer.WeightGradients
                };

                backpropigationHandle =
                    accumulateGradientOverWeightJob.Schedule(nextLayer.WeightGradients.Length, 4,
                        backpropigationHandle);
            }

            JobHandle.ScheduleBatchedJobs();
        

            //Update weights (all layers but first)
            JobHandle updateNetworkJobHandle = backpropigationHandle;

            for (int layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                var layer = _layers[layerIndex];

                var applyGradientsToWeightsJob = new ErrorEvaluators.ApplyGradientToLayerWeights()
                {
                    TestCount = 1,
                    LearningRate = LearningRate,
                    WeightGradients = layer.WeightGradients,
                    LayerWeights = layer.ModelLayer.Weights
                };

                updateNetworkJobHandle =
                    applyGradientsToWeightsJob.Schedule(layer.ModelLayer.Weights.Length, 4, updateNetworkJobHandle);

                //updateNetworkJobHandle.Complete();

                var applyGradientsToBiasesJob = new ErrorEvaluators.ApplyGradientToLayerBiases()
                {
                    TestCount = 1,
                    LearningRate = LearningRate,
                    LayerBiases = layer.ModelLayer.Biases,
                    LayerErrors = layer.Error.Slice(0, testCaseCount)
                };

                updateNetworkJobHandle =
                    applyGradientsToBiasesJob.Schedule(layer.ModelLayer.Biases.Length, 4, updateNetworkJobHandle);

                //updateNetworkJobHandle.Complete();
                int x = 1;
            }

            //return error i guess ?

            updateNetworkJobHandle.Complete();

            //Compute average error sum for test cases
            ComputeErrorSum(expectedOutput, testCaseCount, out errorSum);

            resultArray.Dispose();

            Profiler.EndSample();
            return updateNetworkJobHandle;
        }

        private void ComputeErrorSum(NativeSlice2D<float> expectedOutput, int testCaseCount, out float errorSum)
        {
            errorSum = 0.0f;
            for (int x = 0; x < testCaseCount; x++)
            {
                for (int y = 0; y < OutputLayer.OutputActivation.Dimensions.y; y++)
                {
                    errorSum += Math.Abs(OutputLayer.OutputActivation[x, y] - expectedOutput[x, y]);
                }
            }

            errorSum /= testCaseCount;
        }
    }
}
