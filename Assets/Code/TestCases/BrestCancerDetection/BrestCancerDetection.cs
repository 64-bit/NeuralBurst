using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace NeuralBurst.TestCases
{
    /// <summary>
    /// Test case for machine learning, data aquired from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
    /// https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    /// </summary>
    public class BrestCancerDetection : MonoBehaviour
    {
        const int InputAttributeCount = 9;

        public int Epochs = 500;

        public TextAsset Dataset;

        public JobHandle LastDependentJobHandle;

        public int DatasetSize;
        public NativeArray<float> InputAttributes;
        public NativeArray<float> ExpectedResults;

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
                        NeuronCount = 9,
                        NeuronType = ENeruonType.Linear
                    },
                    new LayerParamaters()
                    {
                        LayerType = ELayerType.Hidden,
                        NeuronCount = 30,
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

            var testSetArray = new NativeArray<float>(9, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            var testResultArray = new NativeArray<float>(1, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            float rollingError = 0.0f;

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                //Select training subset
                int index = epoch % ExpectedResults.Length;
                //Upload to input array
                NativeArray<float>.Copy(InputAttributes, index * InputAttributeCount, testSetArray,0, InputAttributeCount);
                //Upload to output array
                testResultArray[0] = ExpectedResults[index];

                //Forward test

                var tempResult = new NativeArray<float>(1,Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                networkEvaluator.Evaluate(testSetArray, tempResult).Complete();

                Debug.Log($"Error:{Math.Abs(tempResult[0] - testResultArray[0])} -- Result:{tempResult[0]} -- Expected:{testResultArray[0]}");
                tempResult.Dispose();

                //Evolve test
                var jobHandle = networkEvaluator.GradientDescentBackpropigate(testSetArray, testResultArray, out float errorSum);

                jobHandle.Complete();

                rollingError = 0.95f * rollingError + 0.05f * errorSum;

                //Print results

                yield return null;
            }

            testSetArray.Dispose();
            testResultArray.Dispose();
        }

        private void ParseDataset()
        {
            //Dataset is in CSV format, with integers numers
            //First index is a sample ID, we can discard that
            //Last index is the class, 2 for benign, 4 for malignant


            var resultLines = Dataset.text.Split(new[] {'\n'}, StringSplitOptions.RemoveEmptyEntries);
            DatasetSize = resultLines.Length;

            InputAttributes = new NativeArray<float>(resultLines.Length * InputAttributeCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            ExpectedResults = new NativeArray<float>(resultLines.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            int inputArrayPtr = 0;
            int resultArrayPtr = 0;

            foreach (var line in resultLines)
            {
                if (line.Length < 10)
                {
                    continue;
                }

                var lineAttributes = line.Split(',').Select((x)=>
                {
                    if (x == "?")
                    {
                        return 5.0f;
                    }
                    try
                    {
                        return (float) Int32.Parse(x);
                    }
                    catch (Exception e)
                    {
                        Debug.Log($"{x}::{e}");
                    }

                    return 0.0f;
                }
                    
                    ).ToArray();

              
                //Skip first
                for (int i = 1; i < 10; i++)
                {
                    InputAttributes[inputArrayPtr++] = lineAttributes[i];
                }
                ExpectedResults[resultArrayPtr++] = Math.Abs(lineAttributes[10] - 2.0f) < 0.0001f ? 0.0f : 1.0f;
            }


        }

        void OnDestroy()
        {
            LastDependentJobHandle.Complete();
            InputAttributes.Dispose();
            ExpectedResults.Dispose();
        }
    }
}
