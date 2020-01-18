using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Assets.Code.NeuralNetworks.NeuronEvaluators;
using NeuralNetworks.Utilities;
using Unity.Collections;
using Unity.Jobs;

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

            return EvaluateInternal(inputData, outputArray);
        }

        private JobHandle EvaluateInternal(NativeArray<float> inputData, NativeArray<float> outputArray)
        {
            //Copy to input layer
            InputLayer.Values.CopyFrom(inputData);
            var jobHandle = inputData.CopyToJob(InputLayer.Values, _lastDependentJobHandle);

            for (int i = 1; i < _layers.Count; i++)
            {
                //Evaluate layers
                jobHandle = EvaluateLayer(1, jobHandle);
            }

            _lastDependentJobHandle = jobHandle;

            //Copy to output
            return OutputLayer.Values.CopyToJob(outputArray, jobHandle);
        }

        private JobHandle EvaluateLayer(int layerIndex, JobHandle lastJobHandle)
        {
            var currentLayer = _layers[layerIndex];
            var lastLayer = _layers[layerIndex - 1];

            var evaluator = NeuronEvaluators.GetEvaluatorForNeuronType(currentLayer.TargetLayer.NeuronType);

            evaluator.SetArguments(lastLayer.Values, currentLayer.Values, currentLayer.TargetLayer.Weights);

            return evaluator.Schedule(lastJobHandle);
        }

        private class EvaluatorLayer
        {
            public readonly int Size;
            public NetworkLayer TargetLayer;
            public NativeArray<float> Values;

            public EvaluatorLayer(NetworkLayer targetLayer)
            {
                Size = targetLayer.Size;
                TargetLayer = targetLayer;
                Values = new NativeArray<float>(targetLayer.Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            }

            public void Dispose()
            {
                Values.Dispose();
            }
        }



    }
}
