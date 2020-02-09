using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace NeuralBurst.TestCases
{
    /// <summary>
    /// Test case for machine learning, data aquired from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
    /// https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    /// </summary>
    public class BrestCancerDetection : MonoBehaviour
    {
        public const int InputAttributeCount = 9;

        public int Epochs = 10;

        public TextAsset DatasetSource;

        private BrestCancerDataset _dataset;

        void Start()
        {
            ParseDataset();
            StartCoroutine(TrainingProcess());
        }

        private IEnumerator TrainingProcess()
        {
            var neuralNetworkConfig = new NetworkDescription()
            {
                Layers = new List<LayerParamaters>()
                {
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Input,
                        NeuronCount = InputAttributeCount,
                        NeuronType = ENeruonType.Input
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 64,
                        NeuronType = ENeruonType.Sigmoid
                    },
                   new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 16,
                        NeuronType = ENeruonType.Sigmoid
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Output,
                        NeuronCount = 1,
                        NeuronType = ENeruonType.Sigmoid
                    },
                }
            };

            var neuralNetwork = new NeuralNetwork(neuralNetworkConfig);
            neuralNetwork.InitNetworkWithRandomValues(-0.05f, 0.05f);

            var networkEvaluator = new NetworkEvaluator(neuralNetwork);


            var tempResult = new NativeArray2D<float>(1,1);

            //Test inital accuracy
            TestInitialAccuracy(networkEvaluator, tempResult.Slice());

            //Learn over a number of iterations
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                //For each epoch, perform training
                for (int tc = 0; tc < _dataset.TrainingSetSize; tc++)
                {
                    _dataset.GetTrainingCase(tc, out var trainingInput, out var trainingResult);

                    //Evolve network
                    var jobHandle = networkEvaluator.GradientDescentBackpropigate(trainingInput, trainingResult,1, out _);
                    jobHandle.Complete();
                }

                float totalError = 0.0f;
                int totalCorrect = 0;

                for (int i = 0; i < _dataset.TestingSetSize; i++)
                {
                    _dataset.GetTestCase(i, out var trainingInput, out var trainingResult);
                    networkEvaluator.Evaluate(trainingInput, tempResult.Slice(), 1).Complete();

                    var error = Math.Abs(tempResult[0,0] - trainingResult[0, 0]);
                    bool wasCorrect = error < 0.5f;
                    totalCorrect += wasCorrect ? 1 : 0;
                    totalError += error;
                }

                float averageError = totalError / (float) _dataset.TestingSetSize;
                float accuracy = (float) totalCorrect / (float) _dataset.TestingSetSize;

                //Forward test
                Debug.Log($"Epoch {epoch}: Accuracy:{accuracy:P2}  Average Error:{averageError:F4}");

                yield return null;
            }

            tempResult.BackingStore.Dispose();
        }

        private void TestInitialAccuracy(NetworkEvaluator networkEvaluator, NativeSlice2D<float> tempResult)
        {
            float totalError = 0.0f;
            int totalCorrect = 0;

            for (int i = 0; i < _dataset.TestingSetSize; i++)
            {
                _dataset.GetTestCase(i, out var trainingInput, out var trainingResult);
                networkEvaluator.Evaluate(trainingInput, tempResult, 1).Complete();

                //var error = Math.Abs(tempResult[0] - trainingResult[0, 0]);
                var error = CrossEntropyCost(tempResult[0,0], trainingResult[0, 0]);
                bool wasCorrect = error < 0.5f;
                totalCorrect += wasCorrect ? 1 : 0;
                totalError += error;
            }

            float averageError = totalError / (float) _dataset.TestingSetSize;
            float accuracy = (float) totalCorrect / (float) _dataset.TestingSetSize;

            //Forward test
            Debug.Log($"Initial: Accuracy:{accuracy:P2}  Average Error:{averageError:F4}");
        }

        private float CrossEntropyCost(float actual, float expected)
        {
            float intermediate = -expected * math.log(actual) - (1.0f - expected) * math.log(1.0f - actual);
            if (float.IsNaN(intermediate))
            {
                intermediate = 0.0f;
            }

            return intermediate;
        }

        private void ProcessEpoch()
        {

        }

        private void ParseDataset()
        {
            _dataset = new BrestCancerDataset(DatasetSource.text);
        }

        void OnDestroy()
        {

        }
    }
}
