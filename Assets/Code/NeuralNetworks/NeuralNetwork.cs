using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks.Utilities;
using Unity.Collections;

namespace NeuralBurst
{
    /// <summary>
    /// Inital Neural Network class.
    /// </summary>
    /// <remarks>
    /// This class is designed for ease of development and training, in the future another class will exist for raw runtime performance and memory usage in a game or other realtime context
    /// </remarks>
    public class NeuralNetwork
    {
        private NetworkLayer[] _layers;
        public IReadOnlyCollection<NetworkLayer> Layers => _layers;

        /// <summary>
        /// Construct a neural network from a <see cref="NetworkDescription"/>
        /// </summary>
        public NeuralNetwork(NetworkDescription description)
        {
            //Ensure we can construct a nerual network from this description
            ValidateLayers(description);

            ConstructLayers(description);
        }

        //Destroy all allocated native resources
        public void Dispose()
        {
            if (_layers != null)
            {
                foreach (var layer in _layers)
                {
                    layer.Dispose();
                }
                _layers = null;
            }
        }

        public int InputSize
        {
            get
            {
                if (_layers == null)
                {
                    throw new InvalidOperationException();
                }

                return _layers[0].Size;
            }
        }

        public int OutputSize
        {
            get
            {
                if (_layers == null)
                {
                    throw new InvalidOperationException();
                }

                return _layers[0].Size;
            }
        }

        /// <summary>
        /// Evaluate the network in a manner that blocks the main thread rather badly
        /// </summary>
        public NativeArray<float> EvaluateNetworkSyncronous(NativeArray<float> input)
        {
            if (input.Length != InputSize)
            {
                throw new ArgumentOutOfRangeException(nameof(input), "The length of the input array must match the length of the input layer");
            }

            throw new NotImplementedException();
        }

        private void ValidateLayers(NetworkDescription description)
        {
            if (description.Layers == null || description.Layers.Count < 2)
            {
                throw new ArgumentException("A neural network must have at least 2 layers (one input and one output)");
            }

            if (description.Layers[0].LayerType != ELayerType.Input)
            {
                throw new ArgumentException("The first layer of a neural network must be a input layer", nameof(description));
            }

            if (description.Layers[description.Layers.Count - 1].LayerType != ELayerType.Output)
            {
                throw new ArgumentException("The last layer of a neural network must be a output layer", nameof(description));
            }
        }

        public void InitNetworkWithRandomValues(float min, float max, uint seed = 1337)
        {
            for(int i = 1; i < _layers.Length;i++)
            {
                var layer = _layers[i];
                NativeArrayUtilities.InitNativeArrayRandomly(layer.Weights, min, max, seed++).Complete();
                NativeArrayUtilities.InitNativeArrayRandomly(layer.Biases, min, max, seed++).Complete();
            }
        }

        private void ConstructLayers(NetworkDescription description)
        {
            _layers = new NetworkLayer[description.Layers.Count];
            for (int i = 0; i < description.Layers.Count; i++)
            {
                if (i == 0)
                {
                    _layers[i] = new NetworkLayer(description.Layers[i]);
                }
                else
                {
                    _layers[i] = new NetworkLayer(description.Layers[i], description.Layers[i-1]);
                }
            }
        }
    }
}