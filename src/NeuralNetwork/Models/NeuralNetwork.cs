
using System.Security.Cryptography;

namespace NeuralNetwork.Models
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        
        public List<Layer> Layers { get; }


        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(List<float> inputSignals)
        {
            if (inputSignals.Count != Topology.InputCount)
            {
                throw new ArgumentException("Wrong input signals count. Not equals input neurons count.");
            }
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if(Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last()
                    .Neurons
                    .OrderByDescending(n => n.Output)
                    .First();
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals();
                var layer = Layers[i];

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }

            }
        }

        private void SendSignalsToInputNeurons(List<float> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<float>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            var lastLayer = Layers.Last();
            var hiddenLayer = CreateLayer(Topology.OutputCount, lastLayer.Count, NeuronType.Output);
            Layers.Add(hiddenLayer);
        }

        private void CreateHiddenLayers()
        {
            for(int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var lastLayer = Layers.Last();
                var hiddenLayer = CreateLayer(Topology.HiddenLayers[j], lastLayer.Count);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer()
        {
            var inputLayer = CreateLayer(Topology.InputCount, 1, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private Layer CreateLayer(
            int neuronsCount, 
            int lastLayerNeuronsCount,
            NeuronType type = NeuronType.Normal)
        {
            var neurons = new List<Neuron>(neuronsCount);
            for (int i = 0; i < neuronsCount; i++)
            {
                var neuron = new Neuron(lastLayerNeuronsCount, type);
                neurons.Add(neuron);
            }
            var layer = new Layer(neurons, type);

            return layer;
        }
    }
}
