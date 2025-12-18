package nn.initialization;

import java.util.Random;

public class RandomUniformInitializer implements WeightInitializer {

    private final double min;
    private final double max;
    private final Random random;

    public RandomUniformInitializer() {
        this(-1.0, 1.0);
    }

    public RandomUniformInitializer(double min, double max) {
        this.min = min;
        this.max = max;
        this.random = new Random();
    }

    public RandomUniformInitializer(double min, double max, long seed) {
        this.min = min;
        this.max = max;
        this.random = new Random(seed);
    }

    @Override
    public double[][] initialize(int inputSize, int outputSize) {
        double[][] weights = new double[inputSize][outputSize];
        double range = max - min;

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = min + random.nextDouble() * range;
            }
        }

        return weights;
    }
}
