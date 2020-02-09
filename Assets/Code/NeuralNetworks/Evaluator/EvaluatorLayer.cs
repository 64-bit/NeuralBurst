using System;
using Assets.Code.NeuralNetworks.Evaluator;
using Unity.Collections;

namespace NeuralBurst
{
    public class EvaluatorLayer
    {
        public const int EVALUATOR_BUCKET_SLICE_SIZE = 32;

        public readonly int Size;
        public readonly NetworkLayer ModelLayer;

        public NativeArray2D<float> OutputActivation;
        public NativeArray2D<float> WeightedInput;

        public NativeArray2D<float> Error;//Also is in a sense, the bias gradient ?

        public NativeArray<float> WeightGradients; //This does not need to scale by slice size

        public virtual NativeSlice2D<float> SliceActivations(int start, int count)
        {
            return OutputActivation.Slice(start, count);
        }

        public virtual NativeSlice2D<float> SliceWeightedInputs(int start, int count)
        {
            return WeightedInput.Slice(start, count);
        }

        public virtual NativeSlice2D<float> SliceError(int start, int count)
        {
            return Error.Slice(start, count);
        }

        protected EvaluatorLayer(NetworkLayer modelLayer, int multiLayers)
        {
            Size = modelLayer.Size;
            ModelLayer = modelLayer;
            OutputActivation = new NativeArray2D<float>(EVALUATOR_BUCKET_SLICE_SIZE, modelLayer.Size);
            WeightedInput = new NativeArray2D<float>(EVALUATOR_BUCKET_SLICE_SIZE, modelLayer.Size);
            Error = new NativeArray2D<float>(EVALUATOR_BUCKET_SLICE_SIZE, Size);
            WeightGradients = new NativeArray<float>(ModelLayer.Weights.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        //Only for derrived classes that do their own thing
        protected EvaluatorLayer(NetworkLayer modelLayer)
        {
            Size = modelLayer.Size;
            ModelLayer = modelLayer;
        }

        public static EvaluatorLayer CreateFromLayer(NetworkLayer modelLayer, int multiLayer)
        {
            if (modelLayer.GetType() == typeof(InputLayer))
            {
                return new EvaluatorInputLayer(modelLayer, multiLayer);
            }

            return new EvaluatorLayer(modelLayer, multiLayer);
        }

        public void Dispose()
        {
            //TODO:::::
        }
    }
}