namespace NeuralNetwork.Models
{
    /// <summary>
    /// Description that defines the neural network
    /// </summary>
    public class Topology
    {
        public int InputCount { get; }

        public int OutputCount { get; }

        /// <summary>
        /// Number of neurons on each hidden layer
        /// </summary>
        public List<int> HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayers = new List<int>(layers.Length);
            HiddenLayers.AddRange(layers);
        }
    }
}
