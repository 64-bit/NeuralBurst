﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralBurst;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Assets.Code.NeuralNetworks.NeuronEvaluators
{
    public interface INeuronEvaluator : IJobParallelFor
    {
        void SetArguments(
            NativeArray<float> sourceValues,
            NativeArray<float> destinationActivation,
            NativeArray<float> destinationWeightedInputs,
            NativeArray<float> weights,
            NativeArray<float> biases);

        JobHandle Schedule(JobHandle lastDependentJobHandle);
    }

    public static class NeuronEvaluators
    {
        public static INeuronEvaluator GetEvaluatorForNeuronType(ENeruonType neuronType) //TODO:Remove entire class
        {
            switch (neuronType)
            {

                case ENeruonType.Linear:
                    return new LinearNeuronEvaluator();
                case ENeruonType.RectifiedLinear:
                    return new RectifiedLinearNeuronEvaluator();
                case ENeruonType.Sigmoid:
                    throw new NotImplementedException();
                default:
                    throw new ArgumentOutOfRangeException(nameof(neuronType), neuronType, null);
            }
        }

        [BurstCompile]
        public struct LinearNeuronEvaluator : INeuronEvaluator
        {
            [ReadOnly]
            private NativeArray<float> _sourceValues;
            [ReadOnly]
            private NativeArray<float> _weights;

            private NativeArray<float> _destinationActivation;
            private NativeArray<float> _destinationWeightedInputs;

            public void Execute(int index)
            {
                //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                var startWeight = index * _sourceValues.Length;
                var activation = 0.0f;

                for (int i = 0; i < _sourceValues.Length; i++)
                {
                    activation += _sourceValues[i] * _weights[startWeight + i];
                }

                _destinationWeightedInputs[index] = activation;//For this neuron the values are the same
                _destinationActivation[index] = activation;
            }

            public void SetArguments(NativeArray<float> sourceValues, NativeArray<float> destinationActivation, NativeArray<float> destinationWeightedInputs, NativeArray<float> weights, NativeArray<float> biases)
            {
                _sourceValues = sourceValues;
                _destinationActivation = destinationActivation;

                _weights = weights;
            }

            public JobHandle Schedule(JobHandle lastDependentJobHandle)
            {
                return this.Schedule(_destinationActivation.Length, 4, lastDependentJobHandle);
            }
        }

        [BurstCompile]
        public struct RectifiedLinearNeuronEvaluator : INeuronEvaluator
        {
            [ReadOnly]
            private NativeArray<float> _sourceValues;
            [ReadOnly]
            private NativeArray<float> _weights;

            private NativeArray<float> _destinationActivation;
            private NativeArray<float> _destinationWeightedInputs;


            public void Execute(int index)
            {
                //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                var startWeight = index * _sourceValues.Length;
                var weightedInput = 0.0f;

                for (int i = 0; i < _sourceValues.Length; i++)
                {
                    weightedInput += _sourceValues[i] * _weights[startWeight + i];
                }

                _destinationWeightedInputs[index] = weightedInput;
                _destinationActivation[index] = Math.Max(0.0f, weightedInput);
            }

            public void SetArguments(NativeArray<float> sourceValues, NativeArray<float> destinationActivation, NativeArray<float> destinationWeightedInputs,  NativeArray<float> weights, NativeArray<float> biases)
            {
                _sourceValues = sourceValues;
                _destinationWeightedInputs = destinationWeightedInputs;
                _destinationActivation = destinationActivation;
                _weights = weights;
            }

            public JobHandle Schedule(JobHandle lastDependentJobHandle)
            {
                return this.Schedule(_destinationActivation.Length, 4, lastDependentJobHandle);
            }
        }

    }
}