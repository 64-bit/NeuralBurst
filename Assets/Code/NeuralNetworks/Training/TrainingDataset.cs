using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace NeuralBurst
{
    public class TrainingDataset
    {
        protected NativeArray<float> InputData;
        protected NativeArray<float> ExpectedResult;


        public int CaseCount { get; protected set; }       
        public int InputSize { get; protected set; }
        public int ResultSize { get; protected set; }

        public int TrainingSetSize { get; protected set; }
        public int TestingSetSize { get; protected set; }

        public JobHandle LastDependentJobHandle;

        protected void InitFromData(int inputSize, int outputSize, int elements, NativeArray<float> inputData, NativeArray<float> results, float trainingSetSize = 0.7f)
        {
            if (inputData.Length != inputSize * elements)
            {
                throw new ArgumentException();       //TODO: 
            }

            if (results.Length != outputSize * elements)
            {
                throw new ArgumentException();
            }

            TrainingSetSize = (int) (elements * trainingSetSize);
            TestingSetSize = elements - TrainingSetSize;

            CaseCount = elements;

            InputSize = inputSize;
            ResultSize = outputSize;

            InputData = inputData;
            ExpectedResult = results;
        }

        public void ShuffleTrainingData()
        {
            var allTrainingInput = GetInputSlice(0, TrainingSetSize);
            var allTrainingResults = GetResultSlice(0, TrainingSetSize);
            uint seed = (uint)UnityEngine.Random.Range(0, 10000000);

            var shuffleJob = new ShuffleTrainingDataJob()
            {
                InputData = allTrainingInput,
                ResultData = allTrainingResults,
                Seed = seed
            };

            LastDependentJobHandle = shuffleJob.Schedule(LastDependentJobHandle);
            JobHandle.ScheduleBatchedJobs();
        }

        public void GetTrainingCase(int trainingCase, out TestDataSlice trainingInput, out TestDataSlice trainingResult)
        {
            if (trainingCase >= TrainingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            trainingInput = GetInputSlice(trainingCase, 1);
            trainingResult = GetResultSlice(trainingCase, 1);
        }

        public void GetTestCase(int testCase, out TestDataSlice testInput, out TestDataSlice testResult)
        {
            if (testCase >= TestingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            testInput = GetInputSlice(TrainingSetSize + testCase, 1);
            testResult = GetResultSlice(TrainingSetSize + testCase, 1);
        }

        private TestDataSlice GetInputSlice(int start, int count)
        {
            return new TestDataSlice()
            {
                Data = InputData.Slice(start * InputSize, count * InputSize),
                ElementsPerSet = InputSize,
                SetCount = count
            };
        }

        private TestDataSlice GetResultSlice(int start, int count)
        {
            return new TestDataSlice()
            {
                Data = ExpectedResult.Slice(start * ResultSize, count * ResultSize),
                ElementsPerSet = ResultSize,
                SetCount = count
            };
        }

        [BurstCompile]
        private struct ShuffleTrainingDataJob : IJob
        {
            public uint Seed;
            public TestDataSlice InputData;
            public TestDataSlice ResultData;

            public void Execute()
            {
                var random = new Unity.Mathematics.Random(Seed);

                for (int i = 0; i < InputData.SetCount; i++)
                {
                    var swap = random.NextInt(0, InputData.SetCount);
                    Swap(InputData,i, swap);
                    Swap(ResultData, i, swap);
                }
            }

            private void Swap(TestDataSlice slice, int a, int b)
            {
                for (int i = 0; i < slice.ElementsPerSet; i++)
                {
                    var tmp = slice[a, i];
                    slice[a, i] = slice[b,i];
                    slice[b, i] = tmp;
                }
            }

        }

    }

    public struct TestDataSlice
    {
        public int SetCount;
        public int ElementsPerSet;

        public NativeSlice<float> Data;

        public float this[int set, int element]

        {
            get => Data[set * ElementsPerSet + element];
            set => Data[set * ElementsPerSet + element] = value;
        }
       
    }
}
