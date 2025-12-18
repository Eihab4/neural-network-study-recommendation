package nn.loss;

public class MeanSquaredError implements LossFunction {

    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                    "Arrays must have same length: " + predicted.length + " vs " + expected.length);
        }

        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - expected[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    @Override
    public double[] gradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                    "Arrays must have same length: " + predicted.length + " vs " + expected.length);
        }

        double[] grad = new double[predicted.length];
        double scale = 2.0 / predicted.length;
        for (int i = 0; i < predicted.length; i++) {
            grad[i] = scale * (predicted[i] - expected[i]);
        }
        return grad;
    }
}
