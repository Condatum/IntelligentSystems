using System;
using System.Collections.Generic;

namespace CodingBackProp
{
  /// <summary>
  /// A fully connected feed-forward network with one hidden layer:
  /// tanh hidden nodes, softmax output nodes, trained by online
  /// (per-item) back-propagation with momentum.
  /// <para>
  /// This variant uses the <b>cross-entropy</b> loss. Combined with softmax,
  /// the output-node signal simplifies to (target - output).
  /// It also applies <b>weight decay</b> (L2 regularization) to the weights;
  /// biases are not decayed.
  /// </para>
  /// </summary>
  public class NeuralNetwork
  {
    // Initial weights and biases are drawn from [InitLow, InitHigh].
    private const double InitLow = 0.0001;
    private const double InitHigh = 0.001;

    private readonly int numInput;
    private readonly int numHidden;
    private readonly int numOutput;

    private readonly double[] inputs;       // most recent input vector
    private readonly double[][] ihWeights;  // input-to-hidden weights [input][hidden]
    private readonly double[] hBiases;      // hidden biases
    private readonly double[] hOutputs;     // hidden activations (tanh)
    private readonly double[][] hoWeights;  // hidden-to-output weights [hidden][output]
    private readonly double[] oBiases;      // output biases
    private double[] outputs;               // output probabilities (softmax)

    // Scratch "signals": the part of each gradient that has no input term.
    private readonly double[] oSignals;
    private readonly double[] hSignals;

    // Previous weight/bias changes, used by the momentum term.
    private readonly double[][] ihPrevDelta;
    private readonly double[] hbPrevDelta;
    private readonly double[][] hoPrevDelta;
    private readonly double[] obPrevDelta;

    private readonly Random rnd = new Random(0); // initial weights + training order

    private double weightDecay; // L2 strength (lambda), set by Train and used by Error

    /// <summary>Creates a network with small random weights and biases.</summary>
    /// <param name="numInput">Number of input nodes (features).</param>
    /// <param name="numHidden">Number of hidden nodes.</param>
    /// <param name="numOutput">Number of output nodes (classes).</param>
    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      inputs = new double[numInput];
      ihWeights = MakeMatrix(numInput, numHidden);
      hBiases = new double[numHidden];
      hOutputs = new double[numHidden];
      hoWeights = MakeMatrix(numHidden, numOutput);
      oBiases = new double[numOutput];

      oSignals = new double[numOutput];
      hSignals = new double[numHidden];
      ihPrevDelta = MakeMatrix(numInput, numHidden);
      hbPrevDelta = new double[numHidden];
      hoPrevDelta = MakeMatrix(numHidden, numOutput);
      obPrevDelta = new double[numOutput];

      var initial = new double[NumWeights];
      for (int i = 0; i < initial.Length; ++i)
        initial[i] = (InitHigh - InitLow) * rnd.NextDouble() + InitLow;
      SetWeights(initial);
    }

    /// <summary>Total number of weights and biases in the network.</summary>
    public int NumWeights =>
      (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;

    // ------------------------------------------------------------------
    // Weights and biases as a flat array
    // ------------------------------------------------------------------

    /// <summary>
    /// Loads all weights and biases from a flat array laid out as:
    /// input-hidden weights, hidden biases, hidden-output weights, output biases.
    /// </summary>
    public void SetWeights(double[] weights)
    {
      if (weights.Length != NumWeights)
        throw new ArgumentException("Bad weights array in SetWeights", nameof(weights));

      int k = 0;
      foreach (double[] block in ParameterBlocks())
        for (int i = 0; i < block.Length; ++i)
          block[i] = weights[k++];
    }

    /// <summary>Returns all weights and biases as a flat array (same layout as <see cref="SetWeights"/>).</summary>
    public double[] GetWeights()
    {
      var result = new double[NumWeights];
      int k = 0;
      foreach (double[] block in ParameterBlocks())
        for (int i = 0; i < block.Length; ++i)
          result[k++] = block[i];
      return result;
    }

    /// <summary>Yields every parameter array in serialization order.</summary>
    private IEnumerable<double[]> ParameterBlocks()
    {
      foreach (double[] row in ihWeights) yield return row;
      yield return hBiases;
      foreach (double[] row in hoWeights) yield return row;
      yield return oBiases;
    }

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------

    /// <summary>Feeds an input vector through the network.</summary>
    /// <returns>A new array of softmax output probabilities (they sum to 1).</returns>
    public double[] ComputeOutputs(double[] xValues)
    {
      Array.Copy(xValues, inputs, xValues.Length);

      for (int j = 0; j < numHidden; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < numInput; ++i)
          sum += inputs[i] * ihWeights[i][j];
        hOutputs[j] = HyperTan(sum + hBiases[j]);
      }

      var oSums = new double[numOutput];
      for (int k = 0; k < numOutput; ++k)
      {
        for (int j = 0; j < numHidden; ++j)
          oSums[k] += hOutputs[j] * hoWeights[j][k];
        oSums[k] += oBiases[k];
      }

      outputs = Softmax(oSums);
      return (double[])outputs.Clone();
    }

    /// <summary>tanh with a cheap saturation for large |x|.</summary>
    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0;
      if (x > 20.0) return 1.0;
      return Math.Tanh(x);
    }

    /// <summary>Scales raw sums so they are positive and add up to 1.</summary>
    private static double[] Softmax(double[] sums)
    {
      double[] exps = Array.ConvertAll(sums, Math.Exp);
      double total = 0.0;
      foreach (double e in exps) total += e;
      return Array.ConvertAll(exps, e => e / total);
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// <summary>
    /// Trains the network with online back-propagation: items are visited in
    /// a fresh random order every epoch and the weights change after each one.
    /// </summary>
    /// <param name="trainData">Rows of [inputs..., one-hot targets...].</param>
    /// <param name="maxEpochs">Number of passes over the training data.</param>
    /// <param name="learnRate">Step size applied to each gradient.</param>
    /// <param name="momentum">Fraction of the previous change added to each new change.</param>
    /// <param name="weightDecay">L2 strength (lambda): each weight is pulled toward zero by lambda * w.</param>
    /// <returns>The trained weights and biases as a flat array.</returns>
    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate, double momentum, double weightDecay)
    {
      this.weightDecay = weightDecay;
      var xValues = new double[numInput];
      var tValues = new double[numOutput];
      var sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int reportEvery = Math.Max(1, maxEpochs / 10);
      for (int epoch = 1; epoch <= maxEpochs; ++epoch)
      {
        if (epoch % reportEvery == 0 && epoch < maxEpochs)
          Console.WriteLine($"epoch = {epoch}  error = {Error(trainData):F4}");

        Shuffle(sequence);
        foreach (int idx in sequence)
        {
          SplitRow(trainData[idx], xValues, tValues);
          ComputeOutputs(xValues);
          BackPropagate(tValues, learnRate, momentum);
        }
      }
      return GetWeights();
    }

    /// <summary>
    /// One back-propagation step for the item just passed through
    /// <see cref="ComputeOutputs"/>. Indices: i = input, j = hidden, k = output.
    /// Signals are "gradient without the input term"; they follow the
    /// (target - output) sign convention, so changes are added, not subtracted.
    /// Weight decay subtracts lambda * w, the gradient of (lambda / 2) * w^2.
    /// </summary>
    private void BackPropagate(double[] tValues, double learnRate, double momentum)
    {
      // 1. Output-node signals.
      for (int k = 0; k < numOutput; ++k)
        oSignals[k] = OutputSignal(tValues[k], outputs[k]);

      // 2. Hidden-node signals: output signals sent back through the
      //    hidden-output weights, times the tanh derivative.
      for (int j = 0; j < numHidden; ++j)
      {
        double derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]);
        double sum = 0.0;
        for (int k = 0; k < numOutput; ++k)
          sum += oSignals[k] * hoWeights[j][k];
        hSignals[j] = derivative * sum;
      }

      // 3. Update all weights and biases (a bias acts like a weight on a constant 1.0 input).
      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ApplyUpdate(ref ihWeights[i][j], ref ihPrevDelta[i][j],
            (hSignals[j] * inputs[i] - weightDecay * ihWeights[i][j]) * learnRate, momentum);

      for (int j = 0; j < numHidden; ++j)
        ApplyUpdate(ref hBiases[j], ref hbPrevDelta[j],
          hSignals[j] * learnRate, momentum);

      for (int j = 0; j < numHidden; ++j)
        for (int k = 0; k < numOutput; ++k)
          ApplyUpdate(ref hoWeights[j][k], ref hoPrevDelta[j][k],
            (oSignals[k] * hOutputs[j] - weightDecay * hoWeights[j][k]) * learnRate, momentum);

      for (int k = 0; k < numOutput; ++k)
        ApplyUpdate(ref oBiases[k], ref obPrevDelta[k],
          oSignals[k] * learnRate, momentum);
    }

    /// <summary>
    /// Output-node signal for softmax + cross-entropy: the softmax derivative
    /// cancels against the loss derivative, leaving just (target - output).
    /// </summary>
    private static double OutputSignal(double target, double output) =>
      target - output;

    /// <summary>Applies a change plus a momentum fraction of the previous change.</summary>
    private static void ApplyUpdate(ref double param, ref double prevDelta,
      double delta, double momentum)
    {
      param += delta;
      param += prevDelta * momentum;
      prevDelta = delta;
    }

    /// <summary>Fisher-Yates shuffle of the visiting order.</summary>
    private void Shuffle(int[] sequence)
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        (sequence[i], sequence[r]) = (sequence[r], sequence[i]);
      }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// <summary>
    /// Mean cross-entropy error per item, average of -sum(t * ln(y)),
    /// plus the penalty (lambda / 2) * sum(w^2) over all weights (not biases).
    /// </summary>
    private double Error(double[][] data)
    {
      var xValues = new double[numInput];
      var tValues = new double[numOutput];
      double sum = 0.0;

      foreach (double[] row in data)
      {
        SplitRow(row, xValues, tValues);
        double[] yValues = ComputeOutputs(xValues);
        for (int k = 0; k < numOutput; ++k)
          sum += -tValues[k] * Math.Log(yValues[k] + 1.0e-15); // constant avoids ln(0)
      }

      double penalty = 0.0;
      foreach (double[][] matrix in new[] { ihWeights, hoWeights })
        foreach (double[] row in matrix)
          foreach (double w in row)
            penalty += w * w;
      penalty *= 0.5 * weightDecay;

      return (sum / data.Length) + penalty;
    }

    /// <summary>Fraction of items whose largest output matches the largest target (winner takes all).</summary>
    public double Accuracy(double[][] data)
    {
      var xValues = new double[numInput];
      var tValues = new double[numOutput];
      int numCorrect = 0;

      foreach (double[] row in data)
      {
        SplitRow(row, xValues, tValues);
        if (MaxIndex(ComputeOutputs(xValues)) == MaxIndex(tValues))
          ++numCorrect;
      }
      return (double)numCorrect / data.Length;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// <summary>Index of the largest value (the first one wins ties).</summary>
    public static int MaxIndex(double[] vector)
    {
      int best = 0;
      for (int i = 1; i < vector.Length; ++i)
        if (vector[i] > vector[best])
          best = i;
      return best;
    }

    /// <summary>Copies a data row into its input part and its target part.</summary>
    private void SplitRow(double[] row, double[] xValues, double[] tValues)
    {
      Array.Copy(row, xValues, numInput);
      Array.Copy(row, numInput, tValues, 0, numOutput);
    }

    /// <summary>Allocates a rows-by-cols matrix of zeros as a jagged array.</summary>
    private static double[][] MakeMatrix(int rows, int cols)
    {
      var result = new double[rows][];
      for (int r = 0; r < rows; ++r)
        result[r] = new double[cols];
      return result;
    }
  }
}
