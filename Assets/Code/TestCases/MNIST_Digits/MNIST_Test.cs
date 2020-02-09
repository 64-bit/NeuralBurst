/*using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace NeuralBurst.TestCases
{
    /// <summary>
    /// Test case for the MNIST handwritten digit test data.
    /// Data courtisty of http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class MNIST_Test : MonoBehaviour
    {
        public TextAsset Labels;
        public TextAsset Images;

        public int Epochs = 50;

        public const int MULTI_LAYER = 16;

        void Start()
        {
            var dataset = new MNIST_Datase(Images.bytes, Labels.bytes);

            var neuralNetworkConfig = new NetworkDescription()
            {
                Layers = new List<LayerParamaters>()
                {
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Input,
                        NeuronCount = 28*28,
                        NeuronType = ENeruonType.Sigmoid
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 256,
                        NeuronType = ENeruonType.Sigmoid
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 32,
                        NeuronType = ENeruonType.Sigmoid
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Output,
                        NeuronCount = 10,
                        NeuronType = ENeruonType.Sigmoid
                    },
                }
            };

            var neuralNetwork = new NeuralNetwork(neuralNetworkConfig);
            neuralNetwork.InitNetworkWithRandomValues(-0.05f, 0.05f);



            StartCoroutine(RunTest(dataset, neuralNetwork));
        }

        private IEnumerator RunTest(MNIST_Datase dataset, NeuralNetwork network)
        {
            var networkEvaluator = new NetworkEvaluator(network);
            var tempResult = new MultiLayerView<float>(10, 1);
            //Test inital accuracy
            TestInitialAccuracy(dataset, networkEvaluator, tempResult);

            //Learn over a number of iterations
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                //For each epoch, perform training
                for (int tc = 0; tc + MULTI_LAYER < dataset.TrainingSetSize; tc += MULTI_LAYER)
                {
                    dataset.GetTrainingCase(tc, MULTI_LAYER, out var trainingInput, out var trainingResult);

                    //Evolve network
                    var jobHandle = networkEvaluator.GradientDescentBackpropigate(trainingInput, trainingResult, out _);
                    jobHandle.Complete();
                }

                float totalError = 0.0f;
                int totalCorrect = 0;

                for (int i = 0; i < dataset.TestingSetSize; i++)
                {
                    dataset.GetTestCase(i, out var trainingInput, out var trainingResult);
                    networkEvaluator.Evaluate(trainingInput, tempResult).Complete();

                    var error = CrossEntropyCost(tempResult, trainingResult);
                    bool wasCorrect = WasCorrect(tempResult, trainingResult);
                    totalCorrect += wasCorrect ? 1 : 0;
                    totalError += error;
                }

                float averageError = totalError / (float)dataset.TestingSetSize;
                float accuracy = (float)totalCorrect / (float)dataset.TestingSetSize;

                //Forward test
                Debug.Log($"Epoch {epoch}: Accuracy:{accuracy:P2}  Average Error:{averageError:F4}");

                yield return null;
            }

            tempResult.Data.Dispose();
        }

        private void TestInitialAccuracy(MNIST_Datase dataset, NetworkEvaluator networkEvaluator, MultiLayerView<float> tempResult)
        {
            float totalError = 0.0f;
            int totalCorrect = 0;

            for (int i = 0; i < dataset.TestingSetSize; i++)
            {
                dataset.GetTestCase(i, out var trainingInput, out var trainingResult);
                networkEvaluator.Evaluate(trainingInput, tempResult).Complete();

                //var error = Math.Abs(tempResult[0] - trainingResult[0, 0]);
                var error = CrossEntropyCost(tempResult, trainingResult);
                bool wasCorrect = WasCorrect(tempResult, trainingResult);
                totalCorrect += wasCorrect ? 1 : 0;
                totalError += error;
            }

            float averageError = totalError / (float)dataset.TestingSetSize;
            float accuracy = (float)totalCorrect / (float)dataset.TestingSetSize;

            //Forward test
            Debug.Log($"Initial: Accuracy:{accuracy:P2}  Average Error:{averageError:F4}");
        }

        private float CrossEntropyCost(MultiLayerView<float> actual, TestDataSlice expected)
        {
            float errorSum = 0.0f;

            for (int i = 0; i < actual.InstanceLength; i++)
            {
                float e = expected[0, i];
                float a = actual[0, i];

                float intermediate = -e * math.log(a) - (1.0f - e) * math.log(1.0f - a);
                if (float.IsNaN(intermediate))
                {
                    intermediate = 0.0f;
                }

                errorSum += intermediate;
            }

            return errorSum;
        }

        private bool WasCorrect(MultiLayerView<float> actual, TestDataSlice expected)
        {
            int maxIndex = -1;
            float maxVal = float.MinValue;

            for (int i = 0; i < actual.InstanceLength; i++)
            {
                if (actual[0, i] > maxVal)
                {
                    maxVal = actual[0, i];
                    maxIndex = i;
                }
            }

            return expected[0, maxIndex] > 0.5f;
        }


    }
}*/