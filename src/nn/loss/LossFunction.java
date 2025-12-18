package nn.loss;

public interface LossFunction {

    double compute(double[] predicted, double[] expected);

    double[] gradient(double[] predicted, double[] expected);
}
