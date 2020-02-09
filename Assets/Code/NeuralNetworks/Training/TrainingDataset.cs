using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace NeuralBurst
{
    public class TrainingDataset
    {
        protected NativeArray2D<float> InputData;
        protected NativeArray2D<float> ExpectedResult;


        public int CaseCount { get; protected set; }       
        public int InputSize { get; protected set; }
        public int ResultSize { get; protected set; }

        public int TrainingSetSize { get; protected set; }
        public int TestingSetSize { get; protected set; }

        public JobHandle LastDependentJobHandle;

        protected void InitFromData(int inputSize, int outputSize, int elements, NativeArray2D<float> inputData, NativeArray2D<float> results, float trainingSetSize = 0.7f)
        {
            if (inputData.Dimensions.x != elements)
            {
                throw new ArgumentException();       //TODO: 
            }

            if (inputData.Dimensions.y != inputSize)
            {
                throw new ArgumentException();       //TODO: 
            }

            if (results.Dimensions.x != elements)
            {
                throw new ArgumentException();
            }

            if (results.Dimensions.y != outputSize)
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

        public void GetTrainingCase(int trainingCase, out NativeSlice2D<float> trainingInput, out NativeSlice2D<float> trainingResult)
        {
            if (trainingCase >= TrainingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            trainingInput = GetInputSlice(trainingCase, 1);
            trainingResult = GetResultSlice(trainingCase, 1);
        }

        public void GetTrainingCase(int trainingCase, int caseCount, out NativeSlice2D<float> trainingInput, out NativeSlice2D<float> trainingResult)
        {
            if (trainingCase + caseCount >= TrainingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            trainingInput = GetInputSlice(trainingCase, caseCount);
            trainingResult = GetResultSlice(trainingCase, caseCount);
        }

        public void GetTestCase(int testCase, out NativeSlice2D<float> testInput, out NativeSlice2D<float> testResult)
        {
            if (testCase >= TestingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            testInput = GetInputSlice(TrainingSetSize + testCase, 1);
            testResult = GetResultSlice(TrainingSetSize + testCase, 1);
        }

        public void GetTestCase(int testCase, int caseCount, out NativeSlice2D<float> testInput, out NativeSlice2D<float> testResult)
        {
            if (testCase + caseCount >= TestingSetSize)
            {
                throw new ArgumentOutOfRangeException();//TODO:DOCS
            }

            testInput = GetInputSlice(TrainingSetSize + testCase, caseCount);
            testResult = GetResultSlice(TrainingSetSize + testCase, caseCount);
        }

        private NativeSlice2D<float> GetInputSlice(int start, int count)
        {
            return InputData.Slice(start, count);


/*            return new TestDataSlice()
            {
                Data = InputData.Slice(start * InputSize, count * InputSize),
                ElementsPerSet = InputSize,
                SetCount = count
            };*/
        }

        private NativeSlice2D<float> GetResultSlice(int start, int count)
        {
            return ExpectedResult.Slice(start, count);
/*            return new TestDataSlice()
            {
                Data = ExpectedResult.Slice(start * ResultSize, count * ResultSize),
                ElementsPerSet = ResultSize,
                SetCount = count
            };*/
        }

        [BurstCompile]
        private struct ShuffleTrainingDataJob : IJob
        {
            public uint Seed;
            public NativeSlice2D<float> InputData;
            public NativeSlice2D<float> ResultData;

            public void Execute()
            {
                var random = new Unity.Mathematics.Random(Seed);

                for (int i = 0; i < InputData.Dimensions.x; i++)
                {
                    var swap = random.NextInt(0, InputData.Dimensions.x);
                    Swap(InputData,i, swap);
                    Swap(ResultData, i, swap);
                }
            }

            private void Swap(NativeSlice2D<float> slice, int a, int b)
            {
                for (int i = 0; i < slice.Dimensions.y; i++)
                {
                    var tmp = slice[a, i];
                    slice[a, i] = slice[b,i];
                    slice[b, i] = tmp;
                }
            }
        }
    }
}