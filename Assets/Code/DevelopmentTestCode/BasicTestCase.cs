using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralBurst.Dev
{
    /// <summary>
    /// This test and entire namespace is just simple test classes used to develop this library
    /// </summary>
    public class BasicTestCase : MonoBehaviour
    {

        void Start()
        {
            var testValues = new[] {0.0f, 1.0f, 1.5f, 2.0f, 3.0f, 15.5f};
            var softmax = MathTesting.Softmax(testValues);

            string result = "";

            foreach (var f in softmax)
            {
                result += $"{f} -";
            }

            Debug.Log(result);
        }
    }
}
