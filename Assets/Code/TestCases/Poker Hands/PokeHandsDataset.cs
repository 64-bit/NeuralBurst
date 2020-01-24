using System;

using System.Linq;
using Unity.Collections;
using UnityEngine;

namespace NeuralBurst.TestCases

{
    /// <summary>
    /// Test case for machine learning, data aquired from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
    /// Poker Hands dataset: https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    /// </summary>
    public class PokeHandsDataset : TrainingDataset
    {
        public const int Attributes = 10;
        public const int ResultClassSize = 10;

        //Data is in format of CSV as such. see the linka above for more details

        /*
         *7. Attribute Information:
           1) S1 “Suit of card #1”
           Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
           
           2) C1 “Rank of card #1”
           Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
           
           3) S2 “Suit of card #2”
           Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
           
           4) C2 “Rank of card #2”
           Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
           
           5) S3 “Suit of card #3”
           Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
           
           6) C3 “Rank of card #3”
           Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
           
           7) S4 “Suit of card #4”
           Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
           
           8) C4 “Rank of card #4”
           Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
           
           9) S5 “Suit of card #5”
           Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
           
           10) C5 “Rank of card 5”
           Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
           
           11) CLASS “Poker Hand”
           Ordinal (0-9)
           
           0: Nothing in hand; not a recognized poker hand 
           1: One pair; one pair of equal ranks within five cards
           2: Two pairs; two pairs of equal ranks within five cards
           3: Three of a kind; three equal ranks within five cards
           4: Straight; five cards, sequentially ranked with no gaps
           5: Flush; five cards with the same suit
           6: Full house; pair + different rank three of a kind
           7: Four of a kind; four equal ranks within five cards
           8: Straight flush; straight + flush
           9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
         *
         */

        public PokeHandsDataset(string csvSource)
        {
            var resultLines = csvSource.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var DatasetSize = resultLines.Length;

            var InputAttributes = new NativeArray<float>(resultLines.Length * Attributes, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            var ExpectedResults = new NativeArray<float>(resultLines.Length * ResultClassSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

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
                for (int i = 0; i < Attributes; i++)
                {
                    InputAttributes[inputArrayPtr++] = lineAttributes[i];
                }

                WriteLabelOneHot((int)lineAttributes[Attributes], 9, resultArrayPtr++, ExpectedResults);
           
            }

            InitFromData(Attributes, ResultClassSize, DatasetSize, InputAttributes, ExpectedResults, 0.75f);
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