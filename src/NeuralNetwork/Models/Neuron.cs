namespace NeuralNetwork.Models
{
    public class Neuron
    {
        /// <summary>
        /// Weights of neuron
        /// </summary>
        public List<float> Weights { get; }

        /// <summary>
        /// Neuron type
        /// </summary>
        public NeuronType NeuronType { get; }

        /// <summary>
        /// Static output weight
        /// </summary>
        public float Output { get; private set; }

        /// <summary>
        /// Static weight
        /// </summary>
        /// <param name="inputCount">input neuron parameters count</param>
        /// <param name="type">type of neuron</param>
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<float>(inputCount);

            for(int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public float FeedForward(List<float> inputs)
        {
            if(inputs.Count != Weights.Count)
            {
                throw new ArgumentException("Wrong inputs count. Not equals weights count");
            }

            var sum = 0.0F;
            for(int i = 0;i < inputs.Count;i++)
            {
                sum += inputs[i] * Weights[i];
            }


            Output = Sigmoid(sum);
            return Output;

        }

        // F(x) = 1 / (1 + e^(-x))
        private float Sigmoid(float x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));

            return (float)result;
        }

        public override string ToString()
        {
            return Output.ToString(); 
        }
    }
}
