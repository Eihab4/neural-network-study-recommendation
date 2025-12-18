package nn.core;

import nn.loss.LossFunction;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private List<Layer> layers;
    private LossFunction lossFunction;
    private double[] lastOutput;

    public NeuralNetwork(LossFunction lossFunction) {
        this.layers = new ArrayList<>();
        this.lossFunction = lossFunction;
    }

    public void addLayer(Layer layer) {
        if (!layers.isEmpty()) {
            Layer lastLayer = layers.get(layers.size() - 1);
            if (lastLayer.getOutputSize() != layer.getInputSize()) {
                throw new IllegalArgumentException(
                        "Layer input size (" + layer.getInputSize() +
                                ") doesn't match previous layer output size (" +
                                lastLayer.getOutputSize() + ")");
            }
        }
        layers.add(layer);
    }

    public double[] forward(double[] input) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        if (input.length != layers.get(0).getInputSize()) {
            throw new IllegalArgumentException(
                    "Input size (" + input.length +
                            ") doesn't match first layer input size (" +
                            layers.get(0).getInputSize() + ")");
        }

        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }

        lastOutput = current.clone();
        return current;
    }

    public double backward(double[] expected, double learningRate) {
        if (lastOutput == null) {
            throw new IllegalStateException("Must call forward() before backward()");
        }

        if (expected.length != lastOutput.length) {
            throw new IllegalArgumentException(
                    "Expected size (" + expected.length +
                            ") doesn't match output size (" + lastOutput.length + ")");
        }

        double[] gradient = lossFunction.gradient(lastOutput, expected);

        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient, learningRate);
        }

        return lossFunction.compute(lastOutput, expected);
    }

    public double train(double[] input, double[] expected, double learningRate) {
        forward(input);
        return backward(expected, learningRate);
    }

    public double[] predict(double[] input) {
        return forward(input);
    }

    public double getLoss(double[] predicted, double[] expected) {
        return lossFunction.compute(predicted, expected);
    }

    public int getLayerCount() {
        return layers.size();
    }

    public Layer getLayer(int index) {
        return layers.get(index);
    }
}
