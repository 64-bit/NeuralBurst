using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Assets.Code.NeuralNetworks.NeuronEvaluators;
using NeuralNetworks.Utilities;
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
        private readonly List<EvaluatorLayer> _layers = new List<EvaluatorLayer>();

        private EvaluatorLayer InputLayer => _layers[0];
        private EvaluatorLayer OutputLayer => _layers[_layers.Count - 1];
        private JobHandle _lastDependentJobHandle;

        public NetworkEvaluator(NeuralNetwork network)
        {
            foreach (var layer in network.Layers)
            {
                _layers.Add(new EvaluatorLayer(layer));
            }
        }

        public void Dispose()
        {
            foreach (var layer in _layers)
            {
                layer.Dispose();
            }
        }

        public JobHandle Evaluate(NativeArray<float> inputData, NativeArray<float> outputArray)
        {
            Profiler.BeginSample("NetworkEvaluator::Evaluate");
            var inputLayer = _layers[0];
            var outputLayer = _layers[_layers.Count - 1];

            if (inputData.Length != inputLayer.Size)
            {
                throw new ArgumentException();//TODO:
            }

            if (outputArray.Length != outputLayer.Size)
            {
                throw new ArgumentException();//TODO:
            }

            var result =  EvaluateInternal(inputData, outputArray);
            JobHandle.ScheduleBatchedJobs();
            Profiler.EndSample();
            return result;
        }

        private JobHandle EvaluateInternal(NativeArray<float> inputData, NativeArray<float> outputArray)
        {
            //Copy to input layer
            InputLayer.OutputActivation.CopyFrom(inputData);
            var jobHandle = inputData.CopyToJob(InputLayer.OutputActivation, _lastDependentJobHandle);

            for (int i = 1; i < _layers.Count; i++)
            {
                //Evaluate layers
                jobHandle = EvaluateLayer(i, jobHandle);
            }

            _lastDependentJobHandle = jobHandle;

            //Copy to output
            return OutputLayer.OutputActivation.CopyToJob(outputArray, jobHandle);
        }

        private JobHandle EvaluateLayer(int layerIndex, JobHandle lastJobHandle)
        {
            var currentLayer = _layers[layerIndex];
            var lastLayer = _layers[layerIndex - 1];

            var evaluator = NeuronEvaluators.GetEvaluatorForNeuronType(currentLayer.TargetLayer.NeuronType);

            evaluator.SetArguments(lastLayer.OutputActivation, currentLayer.OutputActivation, currentLayer.WeightedInput, currentLayer.TargetLayer.Weights, currentLayer.TargetLayer.Biases);

            var handle =  evaluator.Schedule(lastJobHandle);
            //handle.Complete();

            return handle;
        }

        //Oh Boy
        public JobHandle GradientDescentBackpropigate(NativeArray<float> inputData, NativeArray<float> expectedOutput, out float errorSum)
        {
            Profiler.BeginSample("NetworkEvaluator::GradientDescentBackpropigate");
            var resultArray = new NativeArray<float>(OutputLayer.Size, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var feedforwardHandle = Evaluate(inputData, resultArray);
            //Now we need to use all the jobs created before to evaluate this.

            //Compute output error

            var computeOutputErrorJob = new ErrorEvaluators.QuadraticSigmoidOutputErrorEvaluator()
            {
                Expected = expectedOutput,
                Actuall = resultArray,
                WeightedActivation = OutputLayer.WeightedInput,
                ErrorOut = OutputLayer.Error
            };

            var outputErrorHandle = computeOutputErrorJob.Schedule(OutputLayer.Error.Length, 4, feedforwardHandle);
            //outputErrorHandle.Complete();
            //Perform backpropigation
            JobHandle backpropigationHandle = outputErrorHandle;

            for (int layerIndex = _layers.Count - 2; layerIndex >= 0; layerIndex--)
            {
                var targetLayer = _layers[layerIndex];
                var nextLayer = _layers[layerIndex + 1];

                if(layerIndex != 0)
                {
                    var backPropigateLayerJob = new ErrorEvaluators.SigmoidLayerErrorEvaluator()
                    {
                        NextLayerWeights = nextLayer.TargetLayer.Weights,
                        NextLayerError = nextLayer.Error,
                        WeightedActivation = targetLayer.WeightedInput,
                        ErrorOutput = targetLayer.Error
                    };

                    backpropigationHandle =
                        backPropigateLayerJob.Schedule(targetLayer.Error.Length, 4, backpropigationHandle);
                    //backpropigationHandle.Complete();
                }

                var accumulateGradientOverWeightJob = new ErrorEvaluators.AccumulateGradientOverWeight()
                {
                    PreviousActivation = targetLayer.OutputActivation,
                    NextError = nextLayer.Error,

                    WeightGradients = nextLayer.WeightGradients
                };

                backpropigationHandle =
                    accumulateGradientOverWeightJob.Schedule(nextLayer.WeightGradients.Length, 4,
                        backpropigationHandle);

                //backpropigationHandle.Complete();
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
                    LearningRate = 0.1f,
                    WeightGradients = layer.WeightGradients,
                    LayerWeights = layer.TargetLayer.Weights
                };

                updateNetworkJobHandle =
                    applyGradientsToWeightsJob.Schedule(layer.TargetLayer.Weights.Length, 4, updateNetworkJobHandle);

                //updateNetworkJobHandle.Complete();

                var applyGradientsToBiasesJob = new ErrorEvaluators.ApplyGradientToLayerBiases()
                {
                    TestCount = 1,
                    LearningRate = 0.1f,
                    LayerBiases = layer.TargetLayer.Biases,
                    LayerErrors = layer.Error
                };

                updateNetworkJobHandle =
                    applyGradientsToBiasesJob.Schedule(layer.TargetLayer.Biases.Length, 4, updateNetworkJobHandle);

                //updateNetworkJobHandle.Complete();
                int x = 1;
            }

            //return error i guess ?

            updateNetworkJobHandle.Complete();

            errorSum = 0.0f;
            for (int i = 0; i < OutputLayer.OutputActivation.Length; i++)
            {
               // errorSum += OutputLayer.Error[i];

                errorSum += Math.Abs(OutputLayer.OutputActivation[i] - expectedOutput[i]);
            }

            resultArray.Dispose();

            Profiler.EndSample();
            return updateNetworkJobHandle;
        }

        private class EvaluatorLayer
        {
            public readonly int Size;
            public readonly NetworkLayer TargetLayer;

            public NativeArray<float> OutputActivation;
            public NativeArray<float> WeightedInput;

            public NativeArray<float> Error;//Also is in a sense, the bias gradient ?
            public NativeArray<float> WeightGradients;

            public EvaluatorLayer(NetworkLayer targetLayer)
            {
                Size = targetLayer.Size;
                TargetLayer = targetLayer;
                OutputActivation = new NativeArray<float>(targetLayer.Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                WeightedInput = new NativeArray<float>(targetLayer.Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

                Error = new NativeArray<float>(Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                WeightGradients = new NativeArray<float>(TargetLayer.Weights.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            }

            public void Dispose()
            {
                OutputActivation.Dispose();
            }
        }



    }
}
