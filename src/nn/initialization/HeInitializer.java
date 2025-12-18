package nn.initialization;

import java.util.Random;

public class HeInitializer implements WeightInitializer {

    private final Random random;

    public HeInitializer() {
        this.random = new Random();
    }

    public HeInitializer(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public double[][] initialize(int inputSize, int outputSize) {
        double[][] weights = new double[inputSize][outputSize];
        double stddev = Math.sqrt(2.0 / inputSize);

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextGaussian() * stddev;
            }
        }

        return weights;
    }
}
