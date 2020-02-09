using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralBurst;
using Unity.Collections;

namespace NeuralBurst
{
    public class EvaluatorOutputLayer : EvaluatorLayer
    {
        private NativeSlice2D<float> _currentResultTarget;

        public EvaluatorOutputLayer(NetworkLayer modelLayer, int multiLayers) : base(modelLayer)
        {
            //OutputActivation = new NativeArray2D<float>(modelLayer.Size, EVALUATOR_BUCKET_SLICE_SIZE);
            WeightedInput = new NativeArray2D<float>(modelLayer.Size, EVALUATOR_BUCKET_SLICE_SIZE);
            Error = new NativeArray2D<float>(Size, EVALUATOR_BUCKET_SLICE_SIZE);
            WeightGradients = new NativeArray<float>(ModelLayer.Weights.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

/*        public override NativeSlice2D<float> SliceActivations(int start, int count)
        {
            return _currentInput;//TODO:Can this safely ignore start and count ?
        }*/

        public override NativeSlice2D<float> SliceActivations(int start, int count)
        {
            return _currentResultTarget;
        }

        public void SetResultTarget(NativeSlice2D<float> newInput)
        {
            _currentResultTarget = newInput;
        }
    }
}
