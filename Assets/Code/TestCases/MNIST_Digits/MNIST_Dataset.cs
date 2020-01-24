using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Configuration;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralBurst.TestCases
{
    /// <summary>
    /// Loader for nmist digit test data
    /// </summary>
    public class MNIST_Datase : TrainingDataset
    {
        public const int IMAGE_SIZE = 28 * 28;

        public MNIST_Datase(byte[] images, byte[] labels)
        {
            var labelList = new List<byte>();
            var imageList = new List<float>();        

            using (var labelStream = new MemoryStream(labels))
            {
                using (var labelReader = new BinaryReader(labelStream))
                {
                    //throw away values
                    labelReader.ReadInt32();
                    labelReader.ReadInt32();

                    while (labelStream.Position != labelStream.Length)
                    {
                        labelList.Add(labelReader.ReadByte());
                    }
                }
            }

            using (var imageStream = new MemoryStream(images))
            {
                using (var imagerReader = new BinaryReader(imageStream))
                {
                    //throw away values
                    imagerReader.ReadInt32();
                    imagerReader.ReadInt32();
                    imagerReader.ReadInt32();
                    imagerReader.ReadInt32();

                    while (imageStream.Position != imageStream.Length)
                    {
                        imageList.Add((float)imageStream.ReadByte() / 255.0f);
                    }
                }
            }
            int imageCount = imageList.Count / (28 * 28);

            Debug.Log($"Read {labelList.Count} labels and {imageCount} images");

            if (imageCount != labelList.Count)
            {
                throw new InvalidOperationException("Label list and images are not the same length");
            }

            int NEWCOUNT = 10000;
            imageCount = NEWCOUNT;

            var labelData = new NativeArray<float>(imageCount * 10, Allocator.Persistent,NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < imageCount; i++)
            {
                WriteLabelOneHot(labelList[i], 9, i, labelData);
            }

            var imageData = new NativeArray<float>(imageCount * 28 * 28, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            imageData.CopyFrom(imageList.Take(28*28* NEWCOUNT).ToArray());

            InitFromData(IMAGE_SIZE, 10, imageCount, imageData, labelData, 0.9f);
        }

        private static void WriteLabelOneHot(int value, int maxValue, int index, NativeArray<float> target)
        {
            int stride = maxValue + 1;

            int start = index * stride;

            for (int i = 0; i <= maxValue; i++)
            {
                float writeValue = value == i ? 1.0f : 0.0f;
                target[start + i] = writeValue;
            }
        }
    }
}