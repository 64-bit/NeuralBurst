﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralBurst
{
    public class NetworkDescription
    {
        public List<LayerParamaters> Layers;
        //TODO: Network data type, for now assume C# float (FP32)
    }

    public struct LayerParamaters
    {
        public int NeuronCount;
        public ELayerType LayerType;
        public ENeruonType NeuronType;
    }

    public enum ELayerType
    {
        Input,
        Hidden,
        Output
    }

    public enum ENeruonType
    {
        Linear,
        RectifiedLinear,
        Sigmoid
    }

    public static class ENeuronTypeExtensions
    {
        public static bool HasBiases(this ENeruonType type)
        {
            if (type == ENeruonType.Sigmoid)
            {
                return true;
            }
            return false;
        }
    }
}
