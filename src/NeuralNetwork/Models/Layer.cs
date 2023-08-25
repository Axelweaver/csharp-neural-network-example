namespace NeuralNetwork.Models
{
    /// <summary>
    /// Neurons layer
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Neurons
        /// </summary>
        public List<Neuron> Neurons { get; }

        /// <summary>
        /// Neurons count
        /// </summary>
        public int NeuronsCount => Neurons?.Count ?? 0;

        public NeuronType NeuronType { get; }

        /// <summary>
        /// ctor
        /// </summary>
        /// <param name="neurons"></param>
        /// <param name="type"></param>
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            // TODO: check all input neurons for a types

            Neurons = neurons;
            NeuronType = type;
        }

        public List<float> GetSignals()
        {
            var result = new List<float>(NeuronsCount);
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return NeuronType.ToString();
        }
    }
}
