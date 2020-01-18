using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralBurst;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace Assets.Code.NeuralNetworks.NeuronEvaluators
{
    public interface INeuronEvaluator : IJobParallelFor
    {
        void SetArguments(
            NativeArray<float> sourceValues,
            NativeArray<float> destinationValues,
            NativeArray<float> weights);

        JobHandle Schedule(JobHandle lastDependentJobHandle);
    }

    public static class NeuronEvaluators
    {
        public static INeuronEvaluator GetEvaluatorForNeuronType(ENeruonType neuronType)
        {
            switch (neuronType)
            {

                case ENeruonType.Linear:
                    return new LinearNeuronEvaluator();
                case ENeruonType.RectifiedLinear:
                    return new RectifiedLinearNeuronEvaluator();
                default:
                    throw new ArgumentOutOfRangeException(nameof(neuronType), neuronType, null);
            }
        }

        [BurstCompile]
        public struct LinearNeuronEvaluator : INeuronEvaluator
        {
            private NativeArray<float> _sourceValues;
            private NativeArray<float> _destinationValues;
            private NativeArray<float> _weights;

            public void Execute(int index)
            {
                //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                var startWeight = index * _sourceValues.Length;
                var activation = 0.0f;

                for (int i = 0; i < _sourceValues.Length; i++)
                {
                    activation += _sourceValues[i] * _weights[startWeight + i];
                }

                _destinationValues[index] = activation;
            }

            public void SetArguments(NativeArray<float> sourceValues, NativeArray<float> destinationValues, NativeArray<float> weights)
            {
                _sourceValues = sourceValues;
                _destinationValues = destinationValues;
                _weights = weights;
            }

            public JobHandle Schedule(JobHandle lastDependentJobHandle)
            {
                return this.Schedule(_destinationValues.Length, 4, lastDependentJobHandle);
            }
        }

        [BurstCompile]
        public struct RectifiedLinearNeuronEvaluator : INeuronEvaluator
        {
            private NativeArray<float> _sourceValues;
            private NativeArray<float> _destinationValues;
            private NativeArray<float> _weights;

            public void Execute(int index)
            {
                //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                var startWeight = index * _sourceValues.Length;
                var activation = 0.0f;

                for (int i = 0; i < _sourceValues.Length; i++)
                {
                    activation += _sourceValues[i] * _weights[startWeight + i];
                }

                activation = Math.Max(0.0f, activation);

                _destinationValues[index] = activation;
            }

            public void SetArguments(NativeArray<float> sourceValues, NativeArray<float> destinationValues, NativeArray<float> weights)
            {
                _sourceValues = sourceValues;
                _destinationValues = destinationValues;
                _weights = weights;
            }

            public JobHandle Schedule(JobHandle lastDependentJobHandle)
            {
                return this.Schedule(_destinationValues.Length, 4, lastDependentJobHandle);
            }
        }

    }
}
