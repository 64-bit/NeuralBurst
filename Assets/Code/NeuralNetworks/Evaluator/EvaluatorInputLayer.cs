using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralBurst;
using Unity.Collections;

namespace Assets.Code.NeuralNetworks.Evaluator
{
    public class EvaluatorInputLayer : EvaluatorLayer
    {
        private NativeSlice2D<float> _currentInput;

        public EvaluatorInputLayer(NetworkLayer modelLayer, int multiLayers) : base(modelLayer)
        {
            OutputActivation = new NativeArray2D<float>(modelLayer.Size, EVALUATOR_BUCKET_SLICE_SIZE);
            //WeightedInput = new NativeArray2D<float>(modelLayer.Size, EVALUATOR_BUCKET_SLICE_SIZE);
            Error = new NativeArray2D<float>(Size, EVALUATOR_BUCKET_SLICE_SIZE);
            WeightGradients = new NativeArray<float>(ModelLayer.Weights.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public override NativeSlice2D<float> SliceActivations(int start, int count)
        {
            return _currentInput;//TODO:Can this safely ignore start and count ?
        }

        public void SetInput(NativeSlice2D<float> newInput)
        {
            _currentInput = newInput;
        }
    }
}
