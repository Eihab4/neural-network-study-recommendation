package nn.activation;

public class Tanh implements ActivationFunction {

    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double tanh = Math.tanh(x);
        return 1 - tanh * tanh;
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
