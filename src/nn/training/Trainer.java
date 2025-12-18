package nn.training;

import nn.core.NeuralNetwork;
import java.util.Random;

public class Trainer {

    private NeuralNetwork network;
    private TrainingConfig config;
    private double[] lossHistory;

    public Trainer(NeuralNetwork network, TrainingConfig config) {
        this.network = network;
        this.config = config;
    }

    public Trainer(NeuralNetwork network) {
        this(network, new TrainingConfig());
    }

    public double[] train(double[][] inputs, double[][] expected) {
        if (inputs.length != expected.length) {
            throw new IllegalArgumentException(
                    "Inputs and expected arrays must have same length");
        }

        int numSamples = inputs.length;
        int epochs = config.getEpochs();
        lossHistory = new double[epochs];

        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        Random random = new Random();

        for (int epoch = 0; epoch < epochs; epoch++) {

            if (config.isShuffle()) {
                shuffleArray(indices, random);
            }

            double totalLoss = 0;

            for (int i = 0; i < numSamples; i++) {
                int idx = indices[i];
                double loss = network.train(
                        inputs[idx],
                        expected[idx],
                        config.getLearningRate());
                totalLoss += loss;
            }

            double avgLoss = totalLoss / numSamples;
            lossHistory[epoch] = avgLoss;

            if (config.isVerbose() && (epoch + 1) % config.getPrintEvery() == 0) {
                System.out
                        .println("Epoch " + (epoch + 1) + "/" + epochs + " - Loss: " + String.format("%.6f", avgLoss));
            }
        }

        if (config.isVerbose()) {
            System.out.println("Training complete. Final loss: " + String.format("%.6f", lossHistory[epochs - 1]));
        }

        return lossHistory;
    }

    public double evaluate(double[][] inputs, double[][] expected) {
        if (inputs.length != expected.length) {
            throw new IllegalArgumentException(
                    "Inputs and expected arrays must have same length");
        }

        double totalLoss = 0;

        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            totalLoss += network.getLoss(prediction, expected[i]);
        }

        return totalLoss / inputs.length;
    }

    public double[] getLossHistory() {
        return lossHistory != null ? lossHistory.clone() : new double[0];
    }

    private void shuffleArray(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
