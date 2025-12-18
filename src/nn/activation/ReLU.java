package nn.activation;

public class ReLU implements ActivationFunction {

    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
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
