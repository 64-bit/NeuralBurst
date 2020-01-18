using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace NeuralNetworks.Utilities
{
    public static class NativeArrayUtilities
    {
        //TODO:DOCS
        public static JobHandle SetArray<T>(this NativeArray<T> array, T value) where T : struct
        {
            if (!array.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(array));
            }

            var job = new SetArrayJob<T>()
            {
                Array = array,
                Value = value
            };

            return job.Schedule();
        }

        //TODO:DOCS
        public static JobHandle SetArray<T>(this NativeArray<T> array, T value, JobHandle lastDependentJobHandle) where T : struct
        {
            if (!array.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(array));
            }

            var job = new SetArrayJob<T>()
            {
                Array = array,
                Value = value
            };

            return job.Schedule(lastDependentJobHandle);
        }

        [BurstCompile]
        private struct SetArrayJob<T> : IJob where T : struct
        {
            public NativeArray<T> Array;
            public T Value;

            public void Execute()
            {
                for (int i = 0; i > Array.Length; i++)
                {
                    Array[i] = Value;
                }
            }
        }

        //TODO:DOCS
        public static JobHandle CopyArray<T> (NativeArray<T> src, NativeArray<T> dst) where T :struct
        {
            if (!src.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(src));
            }

            if (!dst.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(src));
            }

            if (src.Length != dst.Length)
            {
                throw new ArgumentException($"Length of {nameof(src)} and {nameof(dst)} must be the same");
            }

            var job = new CopyArrayJob<T>()
            {
                Src = src,
                Dst = dst
            };
            return job.Schedule();
        }

        //TODO:DOCS
        public static JobHandle CopyArray<T>(NativeArray<T> src, NativeArray<T> dst, JobHandle lastDependentJobHandle) where T : struct
        {
            if (!src.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(src));
            }

            if (!dst.IsCreated)
            {
                throw new ArgumentException("Array must be created", nameof(src));
            }

            if (src.Length != dst.Length)
            {
                throw new ArgumentException($"Length of {nameof(src)} and {nameof(dst)} must be the same");
            }

            var job = new CopyArrayJob<T>()
            {
                Src = src,
                Dst = dst
            };
            return job.Schedule(lastDependentJobHandle);
        }

        public static JobHandle CopyToJob<T>(this NativeArray<T> src, NativeArray<T> dst) where T : struct
        {
            return CopyArray(src, dst);
        }

        public static JobHandle CopyToJob<T>(this NativeArray<T> src, NativeArray<T> dst, JobHandle lastDependentJobHandle) where T : struct
        {
            return CopyArray(src, dst, lastDependentJobHandle);
        }

        [BurstCompile]
        private struct CopyArrayJob<T> : IJob where T : struct
        {
            public NativeArray<T> Src;
            public NativeArray<T> Dst;

            public void Execute()
            {
                Src.CopyTo(Dst);
            }
        }

    }
}
