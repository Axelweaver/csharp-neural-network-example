namespace NeuralNetwork.Models
{
    public class Neuron
    {
        /// <summary>
        /// Weights of neuron
        /// </summary>
        public List<float> Weights { get; }

        /// <summary>
        /// Input signals
        /// </summary>
        public List<float> Inputs { get; }

        /// <summary>
        /// Neuron type
        /// </summary>
        public NeuronType NeuronType { get; }

        /// <summary>
        /// Static output weight
        /// </summary>
        public float Output { get; private set; }

        /// <summary>
        /// 
        /// </summary>
        public float Delta { get; private set; }

        /// <summary>
        /// ctor
        /// </summary>
        /// <param name="inputCount">input neuron parameters count</param>
        /// <param name="type">type of neuron</param>
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<float>(inputCount);
            Inputs = new List<float>(inputCount);

            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add((float)rnd.NextDouble());
                Inputs.Add(0);
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
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0F;
            for(int i = 0;i < inputs.Count;i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if(NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;

        }

        // F(x) = 1 / (1 + e^(-x))
        private float Sigmoid(float x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));

            return (float)result;
        }

        private float SigmoidDx(float x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        public void SetWeight(params float[] weights)
        {
            if(weights == null)
            {
                throw new ArgumentNullException();
            }

            if(weights.Length != Weights.Count)
            {
                throw new ArgumentException("Wrong arguments count.");
            }

            // TODO: delete after adding network learning capability
            for(int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }

        }

        public void Learn(float error, float learningRate)
        {
            // input neurons not needed to learning
            if(NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for(int i = 0; i < Weights.Count;i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString(); 
        }
    }
}
