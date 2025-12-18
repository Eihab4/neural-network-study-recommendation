package nn.activation;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    }

    @Override
    public double[] activate(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = activate(input[i]);
        }
        return result;
    }

    @Override
    public double[] derivative(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = derivative(input[i]);
        }
        return result;
    }
}
