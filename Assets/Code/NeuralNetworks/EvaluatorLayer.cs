using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;

namespace NeuralBurst
{
    public class EvaluatorLayer
    {
        public readonly int Size;
        public readonly NetworkLayer ModelLayer;

        public NativeArray<float> OutputActivation;
        public NativeArray<float> WeightedInput;

        public NativeArray<float> Error;//Also is in a sense, the bias gradient ?
        public NativeArray<float> WeightGradients;

        public EvaluatorLayer(NetworkLayer modelLayer, int multiLayers)
        {
            Size = modelLayer.Size;
            ModelLayer = modelLayer;
            OutputActivation = new NativeArray<float>(modelLayer.Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            WeightedInput = new NativeArray<float>(modelLayer.Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            Error = new NativeArray<float>(Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            WeightGradients = new NativeArray<float>(ModelLayer.Weights.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose()
        {
            OutputActivation.Dispose();
        }
    }


    public struct MultiLayerView
    {
        public int SetCount;
        public int ElementsPerSet;

        public NativeArray<float> Data;

        public MultiLayerView(int layerSize, int layers)
        {
            ElementsPerSet = layerSize;
            SetCount = layers;
            Data = new NativeArray<float>(layerSize * layers, Allocator.Persistent);
        }

        public float this[int set, int element]

        {
            get => Data[set * ElementsPerSet + element];
            set => Data[set * ElementsPerSet + element] = value;
        }

    }

}
