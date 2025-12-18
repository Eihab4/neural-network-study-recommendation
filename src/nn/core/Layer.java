package nn.core;

import nn.activation.ActivationFunction;
import nn.initialization.WeightInitializer;
import nn.initialization.HeInitializer;
import nn.util.MatrixUtils;

public class Layer {

    private double[][] weights;
    private ActivationFunction activation;

    private double[] lastInputWithBias;
    private double[] lastWeightedSum;
    private double[] lastOutput;

    private int inputSize;
    private int outputSize;

    public Layer(int inputSize, int outputSize, ActivationFunction activation) {
        this(inputSize, outputSize, activation, new HeInitializer());
    }

    public Layer(int inputSize, int outputSize, ActivationFunction activation, WeightInitializer initializer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.weights = initializer.initialize(inputSize + 1, outputSize);
    }

    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                    "Expected input size " + inputSize + ", got " + input.length);
        }

        double[] inputWithBias = new double[inputSize + 1];
        System.arraycopy(input, 0, inputWithBias, 0, inputSize);
        inputWithBias[inputSize] = 1.0;

        this.lastInputWithBias = inputWithBias.clone();

        double[][] inputMatrix = MatrixUtils.toRowMatrix(inputWithBias);
        double[][] weightedSumMatrix = MatrixUtils.multiply(inputMatrix, weights);
        double[] weightedSum = MatrixUtils.flatten(weightedSumMatrix);

        this.lastWeightedSum = weightedSum.clone();
        this.lastOutput = activation.activate(weightedSum);

        return lastOutput.clone();
    }

    public double[] backward(double[] gradient, double learningRate) {
        if (gradient.length != outputSize) {
            throw new IllegalArgumentException(
                    "Expected gradient size " + outputSize + ", got " + gradient.length);
        }

        double[] activationDerivative = activation.derivative(lastWeightedSum);
        double[] activationGradient = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            activationGradient[i] = gradient[i] * activationDerivative[i];
        }

        double[][] inputT = MatrixUtils.toColumnMatrix(lastInputWithBias);
        double[][] activationGradMatrix = MatrixUtils.toRowMatrix(activationGradient);
        double[][] weightGradient = MatrixUtils.multiply(inputT, activationGradMatrix);

        double[][] scaledWeightGrad = MatrixUtils.scale(weightGradient, learningRate);
        weights = MatrixUtils.subtract(weights, scaledWeightGrad);

        double[][] weightsWithoutBias = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            weightsWithoutBias[i] = weights[i].clone();
        }

        double[][] weightsT = MatrixUtils.transpose(weightsWithoutBias);
        double[][] inputGradMatrix = MatrixUtils.multiply(activationGradMatrix, weightsT);
        double[] inputGradient = MatrixUtils.flatten(inputGradMatrix);

        return inputGradient;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public double[][] getWeights() {
        return MatrixUtils.copy(weights);
    }

    public double[] getBiases() {
        double[] biases = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            biases[j] = weights[inputSize][j];
        }
        return biases;
    }
}
