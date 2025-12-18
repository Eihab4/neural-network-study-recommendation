package nn.loss;

public class CrossEntropyLoss implements LossFunction {

    private static final double EPSILON = 1e-15;

    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                    "Arrays must have same length: " + predicted.length + " vs " + expected.length);
        }

        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double p = clip(predicted[i]);
            sum -= expected[i] * Math.log(p) + (1 - expected[i]) * Math.log(1 - p);
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
        for (int i = 0; i < predicted.length; i++) {
            double p = clip(predicted[i]);
            grad[i] = (-expected[i] / p + (1 - expected[i]) / (1 - p)) / predicted.length;
        }
        return grad;
    }

    private double clip(double value) {
        return Math.max(EPSILON, Math.min(1 - EPSILON, value));
    }
}
