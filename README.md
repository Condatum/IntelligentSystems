# IS-Back-Propagation

A small, dependency-free C# demo of a neural network trained with
back-propagation. It comes in two variants that differ only in the loss
function and regularization, so you can compare them side by side.

| Folder         | Loss                      | Output-node signal                | Regularization          |
|----------------|---------------------------|-----------------------------------|-------------------------|
| `CrossEntropy` | Cross-entropy             | `t - y` (softmax derivative cancels) | none                 |
| `WeightDecay`  | Cross-entropy + L2 penalty| `t - y`                           | `lambda = 0.001` on weights (not biases) |

## What the demo does

1. A randomly weighted **teacher** network (4-5-3) labels 1000 random items.
2. The data is shuffled and split 80% train / 20% test.
3. A fresh network is trained on the training set and the final weights and
   the train/test accuracy are printed. Error is printed every 10% of the epochs.

## Network

* Fully connected, one hidden layer, `tanh` hidden nodes, `softmax` outputs.
* Online training: weights change after every item, items visited in random order.
* Momentum: each update also adds `momentum * previous update`.

## Layout

```
<Variant>/
  Program.cs        demo driver: data generation, split, console display
  NeuralNetwork.cs  the network: forward pass, back-propagation, error, accuracy
  <Variant>.csproj  .NET 8 console project
```

## Run

Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download).

```
cd CrossEntropy      # or WeightDecay
dotnet run
```

## Where the two variants differ

`WeightDecay` is `CrossEntropy` plus L2 regularization. The differences live in
`NeuralNetwork.cs`, plus the extra `weightDecay` argument in `WeightDecay/Program.cs`:

* `WeightDecay` adds: `Train(..., weightDecay)`, the `- weightDecay * w` term in
  the weight updates, and the penalty in `Error`.

## Refactoring notes

* Split into `Program.cs` and `NeuralNetwork.cs`; every member has XML docs.
* Gradient arrays are gone: each gradient is computed and applied in place.
* The four near-identical update blocks share one `ApplyUpdate` helper.
* Weights are (de)serialized through one `ParameterBlocks()` iterator instead of
  two hand-written loops.
* Dead and commented-out code removed; `out` parameters replaced by a tuple.
* Math, random-number order and printed output are meant to be unchanged.
  One robustness fix: the error-report interval can no longer be 0 when
  `maxEpochs < 10`.
# IntelligentSystems
