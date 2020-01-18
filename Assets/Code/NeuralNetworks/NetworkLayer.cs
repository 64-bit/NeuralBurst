using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Jobs;

namespace NeuralBurst
{
    public class NetworkLayer
    {
        public NativeArray<float> Weights
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

        public NetworkLayer()
        {
            
        }

        public NetworkLayer(LayerParamaters layerParamaters, LayerParamaters previousLayer)
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
        }

        public NetworkLayer(LayerParamaters layerParamaters)
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

        public void Dispose()
        {
            LastDependentJobHandle.Complete();
            Weights.Dispose();
        }
    }
}