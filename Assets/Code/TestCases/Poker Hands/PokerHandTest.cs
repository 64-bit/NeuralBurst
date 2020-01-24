﻿using System;
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
    /// Test case for classifying poker hands
    /// Data courtisty of https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    /// </summary>
    public class PokerHandTest : MonoBehaviour
    {
        public TextAsset DataCSV;
        public int Epochs = 50;

        void Start()
        {
            var dataset = new PokeHandsDataset(DataCSV.text);

            var neuralNetworkConfig = new NetworkDescription()
            {
                Layers = new List<LayerParamaters>()
                {
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Input,
                        NeuronCount = PokeHandsDataset.Attributes,
                        NeuronType = ENeruonType.Sigmoid
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 128,
                        NeuronType = ENeruonType.RectifiedLinear
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 32,
                        NeuronType = ENeruonType.RectifiedLinear
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Output,
                        NeuronCount = PokeHandsDataset.ResultClassSize,
                        NeuronType = ENeruonType.Sigmoid
                    },
                }
            };

            var neuralNetwork = new NeuralNetwork(neuralNetworkConfig);
            neuralNetwork.InitNetworkWithRandomValues(-0.05f, 0.05f);

            StartCoroutine(RunTest(dataset, neuralNetwork));
        }

        private IEnumerator RunTest(PokeHandsDataset dataset, NeuralNetwork network)
        {
            var networkEvaluator = new NetworkEvaluator(network);
            var tempResult = new NativeArray<float>(10, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //Test inital accuracy
            TestInitialAccuracy(dataset, networkEvaluator, tempResult);

            //Learn over a number of iterations
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                //For each epoch, perform training
                for (int tc = 0; tc < dataset.TrainingSetSize; tc++)
                {
                    dataset.GetTrainingCase(tc, out var trainingInput, out var trainingResult);

                    if (tc % 1000 == 0)
                    {
                        yield return null;
                    }

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

            tempResult.Dispose();
        }

        private void TestInitialAccuracy(PokeHandsDataset dataset, NetworkEvaluator networkEvaluator, NativeArray<float> tempResult)
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

        private float CrossEntropyCost(NativeArray<float> actual, TestDataSlice expected)
        {
            float errorSum = 0.0f;

            for (int i = 0; i < actual.Length; i++)
            {
                float e = expected[0, i];
                float a = actual[i];

                float intermediate = -e * math.log(a) - (1.0f - e) * math.log(1.0f - a);
                if (float.IsNaN(intermediate))
                {
                    intermediate = 0.0f;
                }

                errorSum += intermediate;
            }

            return errorSum;
        }

        private bool WasCorrect(NativeArray<float> actual, TestDataSlice expected)
        {
            int maxIndex = -1;
            float maxVal = float.MinValue;

            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i] > maxVal)
                {
                    maxVal = actual[i];
                    maxIndex = i;
                }
            }

            return expected[0, maxIndex] > 0.5f;
        }
    }

}