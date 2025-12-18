package nn.activation;

public interface ActivationFunction {

    double activate(double x);

    double derivative(double x);

    double[] activate(double[] input);

    double[] derivative(double[] input);
}
