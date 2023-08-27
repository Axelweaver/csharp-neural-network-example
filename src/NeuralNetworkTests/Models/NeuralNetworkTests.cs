using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NeuralNetwork.Models.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new float[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new float[,]
            {
                // Result - the patient is sick     - 1
                //          the patient is healthy  - 0

                // wrong temperature T
                // good age A
                // smoking S
                // good food F
                //T  A  S  F
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };

            var topology = new Topology(4, 1, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            neuralNetwork.Layers[1].Neurons[0].SetWeight(0.5F, -0.1F, 0.3F, -0.1F);
            neuralNetwork.Layers[1].Neurons[1].SetWeight(0.1F, -0.3F, 0.7F, -0.3F);
            neuralNetwork.Layers[2].Neurons[0].SetWeight(1.2F, 0.8F);

            var result = neuralNetwork.FeedForward(new List<float>(4) { 1, 0, 0, 0 });

            Assert.IsTrue(result.Output >= 0.5F);
        }
    }
}