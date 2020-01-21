using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralBurst.TestCases
{
    public class BrestCancerDataset : TrainingDataset
    {

        public BrestCancerDataset(string source, float trainingSetSize = 0.7f)
        {
            //Dataset is in CSV format, with integers numers
            //First index is a sample ID, we can discard that
            //Last index is the class, 2 for benign, 4 for malignant


            var resultLines = source.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var DatasetSize = resultLines.Length;

            var InputAttributes = new NativeArray<float>(resultLines.Length * BrestCancerDetection.InputAttributeCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            var ExpectedResults = new NativeArray<float>(resultLines.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            int inputArrayPtr = 0;
            int resultArrayPtr = 0;

            foreach (var line in resultLines)
            {
                if (line.Length < 10)
                {
                    continue;
                }

                var lineAttributes = line.Split(',').Select((x) =>
                    {
                        if (x == "?")
                        {
                            return 5.0f;
                        }
                        try
                        {
                            return (float)Int32.Parse(x);
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
            InitFromData(BrestCancerDetection.InputAttributeCount, 1, DatasetSize, InputAttributes, ExpectedResults, trainingSetSize);
        }
    }
}
