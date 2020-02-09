using System;
using Unity.Collections;
using Unity.Mathematics;

namespace NeuralBurst
{
    public struct NativeArray2D<T> where T : struct
    {
        public NativeArray<T> BackingStore;
        private int2 _dimensions;
        public int2 Dimensions => _dimensions;

        public NativeArray2D(int2 dimensions)
        {
            _dimensions = dimensions;
            BackingStore = new NativeArray<T>(_dimensions.x * _dimensions.y, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public NativeArray2D(int x, int y)
        {
            _dimensions = new int2(x,y);
            BackingStore = new NativeArray<T>(_dimensions.x * _dimensions.y, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        public T this[int2 position]
        {
            get => BackingStore[GetIndex(position)];
            set => BackingStore[GetIndex(position)] = value;
        }

        public T this[int x, int y]
        {
            get => BackingStore[GetIndex(x,y)];
            set => BackingStore[GetIndex(x,y)] = value;
        }

        public int GetIndex(int2 position)
        {
            return position.x * _dimensions.y + position.y;
        }

        public int GetIndex(int x, int y)
        {
            return x * _dimensions.y + y;      
        }

        public NativeSlice2D<T> Slice(int start, int count)
        {
            return new NativeSlice2D<T>(this, start,count);
        }

        public NativeSlice2D<T> Slice()
        {
            return new NativeSlice2D<T>(this, 0, _dimensions.x);
        }

        public void Dispose()
        {
            BackingStore.Dispose();
        }
    }

    public struct NativeSlice2D<T> where T : struct
    {
        public NativeSlice<T> BackingSlice;
        private int2 _dimensions;
        public int2 Dimensions => _dimensions;

        public NativeSlice2D(NativeArray2D<T> source, int start, int count)
        {
            _dimensions = new int2(count, source.Dimensions.y);

            int sliceOffset = start * _dimensions.y;
            int sliceCount = count * _dimensions.y;

            BackingSlice = source.BackingStore.Slice(sliceOffset, sliceCount);
        }

        public T this[int2 position]
        {
            get => BackingSlice[GetIndex(position)];
            set => BackingSlice[GetIndex(position)] = value;
        }

        public T this[int x, int y]
        {
            get => BackingSlice[GetIndex(x, y)];
            set => BackingSlice[GetIndex(x, y)] = value;
        }

        public int GetIndex(int2 position)
        {
            return position.x * _dimensions.y + position.y;
        }

        public int GetIndex(int x, int y)
        {
            return x * _dimensions.y + y;
        }
    }
}
