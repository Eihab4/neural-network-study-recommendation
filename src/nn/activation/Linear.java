package nn.activation;

public class Linear implements ActivationFunction {

    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }

    @Override
    public double[] activate(double[] input) {
        return input.clone();
    }

    @Override
    public double[] derivative(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = 1;
        }
        return result;
    }
}
