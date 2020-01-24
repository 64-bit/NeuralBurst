using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;

namespace NeuralBurst
{
    public abstract class NetworkLayer //Why not construct this directly, and have this be a virtual class ? 
    {
        public NativeArray<float> Weights
        {
            get;
            private set;
        }

        public NativeArray<float> Biases
        {
            get;
            private set;
        }

        public int PreviousLayerSize
        {
            get; protected set;
        }

        public JobHandle LastDependentJobHandle = default;

        //TODO:Weights ??? or is that part of the main class
        //weights depend on the layer before this, but very much belong to this layer ?

        public int Size
        {
            get;
            protected set;
        }

        public ENeruonType NeuronType
        {
            get;
            protected set;
        }

        public ELayerType LayerType
        {
            get;
            protected set;
        }

        public abstract JobHandle EvaluateLayer(EvaluatorLayer lastLayerState, EvaluatorLayer currentLayerState, JobHandle jobHandleToWaitOn);

        public abstract JobHandle BackpropigateLayer(EvaluatorLayer currentLayerState, EvaluatorLayer nextLayerState,
            JobHandle jobHandleToWaitOn);


        protected void InitLayerInput(LayerParamaters layerParamaters)
        {
            if (layerParamaters.LayerType != ELayerType.Input)
            {
                throw new InvalidOperationException("This constructor can only be used for input layers");
            }

            Size = layerParamaters.NeuronCount;
            NeuronType = layerParamaters.NeuronType;
            LayerType = layerParamaters.LayerType;

            PreviousLayerSize = -1;
        }

        protected void InitLayer(LayerParamaters layerParamaters, LayerParamaters previousLayer)
        {
            if (layerParamaters.LayerType == ELayerType.Input)
            {
                throw new InvalidOperationException("This constructor cannot be used for input layers");
            }

            Size = layerParamaters.NeuronCount;
            NeuronType = layerParamaters.NeuronType;
            LayerType = layerParamaters.LayerType;

            //construct weights array
            PreviousLayerSize = previousLayer.NeuronCount;
            int weightCount = layerParamaters.NeuronCount * previousLayer.NeuronCount;
            Weights = new NativeArray<float>(weightCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Biases = new NativeArray<float>(Size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public static NetworkLayer ConstructLayer(LayerParamaters layerParamaters)
        {
            switch (layerParamaters.NeuronType)
            {
                case ENeruonType.Linear:
                    return new LinearLayer(layerParamaters);
                case ENeruonType.RectifiedLinear:
                    return new RectifiedLinearLayer(layerParamaters);
                case ENeruonType.Sigmoid:
                    return new SigmoidLayer(layerParamaters);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static NetworkLayer ConstructLayer(LayerParamaters layerParamaters, LayerParamaters previousLayer)
        {
            switch (layerParamaters.NeuronType)
            {
                case ENeruonType.Linear:
                    return new LinearLayer(layerParamaters, previousLayer);
                case ENeruonType.RectifiedLinear:
                    return new RectifiedLinearLayer(layerParamaters, previousLayer);
                case ENeruonType.Sigmoid:
                    return new SigmoidLayer(layerParamaters, previousLayer);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public void Dispose()
        {
            LastDependentJobHandle.Complete();
            Weights.Dispose();
        }
    }
}