using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace NeuralBurst
{
    class RectifiedLinearLayer : NetworkLayer
    {
        public RectifiedLinearLayer(LayerParamaters layerParamaters)
        {
            InitLayerInput(layerParamaters);
        }

        public RectifiedLinearLayer(LayerParamaters layerParamaters, LayerParamaters previousLayer)
        {
            InitLayer(layerParamaters, previousLayer);
        }

        public override JobHandle EvaluateLayer(EvaluatorLayer lastLayerState, EvaluatorLayer currentLayerState,int count, JobHandle jobHandleToWaitOn)
        {
            var layerEvaluatorJob = new RectifiedLinearEvaluator()
            {
                SourceActivation = lastLayerState.SliceActivations(0, count),
                LayerWeightedInputs = currentLayerState.SliceWeightedInputs(0, count),
                LayerActivation = currentLayerState.SliceActivations(0, count),

                Weights = Weights,
                Biases = Biases
            };
            return layerEvaluatorJob.Schedule(currentLayerState.OutputActivation.Dimensions.y, 4, jobHandleToWaitOn);
        }

        public override JobHandle BackpropigateLayer(EvaluatorLayer currentLayerState, EvaluatorLayer nextLayerState,int count, JobHandle jobHandleToWaitOn)
        {
            var backPropigateLayerJob = new RectifiedLinearLayerErrorEvaluator()
            {
                NextLayerWeights = nextLayerState.ModelLayer.Weights,
                NextLayerError = nextLayerState.SliceError(0, count),
                WeightedActivation = currentLayerState.SliceWeightedInputs(0, count),
                ErrorOutput = currentLayerState.SliceError(0, count)
            };
            return backPropigateLayerJob.Schedule(currentLayerState.Error.Dimensions.y, 4, jobHandleToWaitOn);
        }

        [BurstCompile]
        private struct RectifiedLinearEvaluator : IJobParallelFor
        {
            [ReadOnly]
            public NativeSlice2D<float> SourceActivation;
            [ReadOnly]
            public NativeArray<float> Weights;
            [ReadOnly]
            public NativeArray<float> Biases;

            public NativeSlice2D<float> LayerActivation;
            public NativeSlice2D<float> LayerWeightedInputs;

            public void Execute(int index)
            {
                for (int view = 0; view < SourceActivation.Dimensions.x; view++)
                {
                    //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                    var startWeight = index * SourceActivation.Dimensions.y;
                    var weightedInput = 0.0f;

                    var bias = Biases[index];

                    for (int i = 0; i < SourceActivation.Dimensions.y; i++)
                    {
                        weightedInput += SourceActivation[view, i] * Weights[startWeight + i];
                    }

                    weightedInput += bias;
                    LayerWeightedInputs[view, index] = weightedInput;

                    var finalActivation = math.max(0.0f, weightedInput);
                    LayerActivation[view, index] = finalActivation;
                }
            }
        }

        [BurstCompile]
        private struct RectifiedLinearLayerErrorEvaluator : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<float> NextLayerWeights;
            [ReadOnly]
            public NativeSlice2D<float> NextLayerError;
            [ReadOnly]
            public NativeSlice2D<float> WeightedActivation;

            public NativeSlice2D<float> ErrorOutput;

            public void Execute(int index)
            {
                for (int view = 0; view < NextLayerError.Dimensions.x; view++)
                {
                    //Multiply the error of every single next node by the weight connecting us to them
                    int nextLayerSize = NextLayerError.Dimensions.y;

                    float backpropigatedError = 0.0f;
                    for (int i = 0; i < nextLayerSize; i++)
                    {
                        float nextNodeError = NextLayerError[view, i];
                        float weightToNextNode = NextLayerWeights[index + i * nextLayerSize];

                        backpropigatedError += nextNodeError * weightToNextNode;
                    }

                    var weightedActivation = WeightedActivation[view, index];
                    //Multiply this by the rate of change of our own activation function
                    float deltaActivation = weightedActivation > 0.0f ? 1.0f : 0.0f;

                    //TODO:May need to accumulate this error, as it is the gradient we apply to the biases
                    ErrorOutput[view, index] = backpropigatedError * deltaActivation;
                }
            }
        }

        /*        [BurstCompile]
                private struct RectifiedLinearEvaluator : IJobParallelFor
                {
                    [ReadOnly]
                    public NativeArray<float> SourceActivation;
                    [ReadOnly]
                    public NativeArray<float> Weights;
                    [ReadOnly]
                    public NativeArray<float> Biases;

                    public NativeArray<float> LayerActivation;
                    public NativeArray<float> LayerWeightedInputs;

                    public void Execute(int index)
                    {
                        //Weights are indexed with all the weights for a given neruon in order, one weight per pervious layer size
                        var startWeight = index * SourceActivation.Length;
                        var weightedInput = 0.0f;

                        var bias = Biases[index];

                        for (int i = 0; i < SourceActivation.Length; i++)
                        {
                            weightedInput += SourceActivation[i] * Weights[startWeight + i];
                        }

                        weightedInput += bias;
                        LayerWeightedInputs[index] = weightedInput;

                        var finalActivation = math.max(0.0f, weightedInput);
                        LayerActivation[index] = finalActivation;
                    }
                }

                [BurstCompile]
                private struct RectifiedLinearLayerErrorEvaluator : IJobParallelFor
                {
                    [ReadOnly]
                    public NativeArray<float> NextLayerWeights;
                    [ReadOnly]
                    public NativeArray<float> NextLayerError;
                    [ReadOnly]
                    public NativeArray<float> WeightedActivation;

                    public NativeArray<float> ErrorOutput;

                    public void Execute(int index)
                    {
                        //Multiply the error of every single next node by the weight connecting us to them
                        int nextLayerSize = NextLayerError.Length;

                        float backpropigatedError = 0.0f;
                        for (int i = 0; i < nextLayerSize; i++)
                        {
                            float nextNodeError = NextLayerError[i];
                            float weightToNextNode = NextLayerWeights[index + i * nextLayerSize];

                            backpropigatedError += nextNodeError * weightToNextNode;
                        }

                        var weightedActivation = WeightedActivation[index];
                        //Multiply this by the rate of change of our own activation function
                        float deltaActivation = weightedActivation > 0.0f ? 1.0f : 0.0f;

                        //TODO:May need to accumulate this error, as it is the gradient we apply to the biases
                        ErrorOutput[index] = backpropigatedError * deltaActivation;
                    }
                }*/
    }
}
