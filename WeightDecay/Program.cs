using System;

namespace CodingBackProp
{
  /// <summary>
  /// Back-propagation demo (cross-entropy loss with weight decay).
  /// <para>
  /// A "teacher" network with random weights labels random data; a second
  /// network is then trained from scratch to reproduce those labels.
  /// </para>
  /// </summary>
  internal static class BackPropProgram
  {
    private static void Main()
    {
      Console.WriteLine("\nBegin neural network back-propagation demo");

      const int numInput = 4;   // number of features
      const int numHidden = 5;
      const int numOutput = 3;  // number of classes
      const int numRows = 1000;
      const int seed = 1;       // gives a nice demo

      Console.WriteLine($"\nGenerating {numRows} artificial data items with {numInput} features");
      double[][] allData = MakeAllData(numInput, numHidden, numOutput, numRows, seed);
      Console.WriteLine("Done");

      Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
      (double[][] trainData, double[][] testData) = SplitTrainTest(allData, 0.80, seed);
      Console.WriteLine("Done\n");

      Console.WriteLine("Training data:");
      ShowMatrix(trainData, 4, 2, true);
      Console.WriteLine("Test data:");
      ShowMatrix(testData, 4, 2, true);

      Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network");
      var nn = new NeuralNetwork(numInput, numHidden, numOutput);

      const int maxEpochs = 1000;
      const double learnRate = 0.05;
      const double momentum = 0.01;
      const double weightDecay = 0.001; // L2 regularization strength
      Console.WriteLine($"\nSetting maxEpochs = {maxEpochs}");
      Console.WriteLine($"Setting learnRate = {learnRate:F2}");
      Console.WriteLine($"Setting momentum  = {momentum:F2}");
      Console.WriteLine($"Setting weightDecay = {weightDecay:F4}");

      Console.WriteLine("\nStarting training");
      double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum, weightDecay);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      ShowVector(weights, 2, 10, true);

      Console.WriteLine($"\nFinal accuracy on training data = {nn.Accuracy(trainData):F4}");
      Console.WriteLine($"Final accuracy on test data     = {nn.Accuracy(testData):F4}");

      Console.WriteLine("\nEnd back-propagation demo\n");
      Console.ReadLine();
    }

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------

    /// <summary>
    /// Builds a labelled data set. Each row is [inputs..., one-hot class...];
    /// the class is the winning output of a randomly weighted "teacher" network.
    /// </summary>
    private static double[][] MakeAllData(int numInput, int numHidden,
      int numOutput, int numRows, int seed)
    {
      var rnd = new Random(seed);
      var teacher = new NeuralNetwork(numInput, numHidden, numOutput);

      var weights = new double[teacher.NumWeights]; // weights and biases
      for (int i = 0; i < weights.Length; ++i)
        weights[i] = RandomInRange(rnd, -10.0, 10.0);

      Console.WriteLine("Generating weights and biases:");
      ShowVector(weights, 2, 10, true);
      teacher.SetWeights(weights);

      var data = new double[numRows][];
      for (int r = 0; r < numRows; ++r)
      {
        var row = new double[numInput + numOutput]; // outputs start at 0.0
        for (int i = 0; i < numInput; ++i)
          row[i] = RandomInRange(rnd, -10.0, 10.0);

        double[] probs = teacher.ComputeOutputs(row[..numInput]);
        row[numInput + NeuralNetwork.MaxIndex(probs)] = 1.0; // 1-of-N encoding
        data[r] = row;
      }
      return data;
    }

    /// <summary>Shuffles the rows (Fisher-Yates) and splits them into train and test sets.</summary>
    private static (double[][] Train, double[][] Test) SplitTrainTest(
      double[][] allData, double trainPct, int seed)
    {
      var rnd = new Random(seed);
      var shuffled = (double[][])allData.Clone(); // copies row references only
      for (int i = 0; i < shuffled.Length; ++i)
      {
        int r = rnd.Next(i, shuffled.Length);
        (shuffled[i], shuffled[r]) = (shuffled[r], shuffled[i]);
      }

      int numTrainRows = (int)(shuffled.Length * trainPct);
      return (shuffled[..numTrainRows], shuffled[numTrainRows..]);
    }

    /// <summary>Uniform random value in [lo, hi).</summary>
    private static double RandomInRange(Random rnd, double lo, double hi) =>
      (hi - lo) * rnd.NextDouble() + lo;

    // ------------------------------------------------------------------
    // Console display
    // ------------------------------------------------------------------

    /// <summary>
    /// Prints the first <paramref name="numRows"/> rows and the last row
    /// (separated by ". . ." when rows are skipped).
    /// </summary>
    private static void ShowMatrix(double[][] matrix, int numRows,
      int decimals, bool indices)
    {
      int width = matrix.Length.ToString().Length;

      void ShowRow(int r)
      {
        if (indices) Console.Write($"[{r.ToString().PadLeft(width)}]  ");
        foreach (double v in matrix[r])
          Console.Write(Format(v, decimals) + "  ");
      }

      for (int r = 0; r < numRows; ++r)
      {
        ShowRow(r);
        Console.WriteLine();
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        ShowRow(matrix.Length - 1);
      }
      Console.WriteLine("\n");
    }

    /// <summary>Prints a vector, <paramref name="lineLen"/> values per line.</summary>
    private static void ShowVector(double[] vector, int decimals,
      int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine();
        Console.Write(Format(vector[i], decimals) + " ");
      }
      if (newLine) Console.WriteLine();
    }

    /// <summary>Fixed-point text with a leading space for non-negative values, so columns line up.</summary>
    private static string Format(double v, int decimals) =>
      (v >= 0.0 ? " " : "") + v.ToString("F" + decimals);
  }
}
