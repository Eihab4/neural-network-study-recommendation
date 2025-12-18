package nn.data;

import java.util.Random;

public class DataSplitter {

    public static class SplitResult {
        public final double[][] trainInputs;
        public final double[][] trainExpected;
        public final double[][] testInputs;
        public final double[][] testExpected;

        public SplitResult(double[][] trainInputs, double[][] trainExpected,
                double[][] testInputs, double[][] testExpected) {
            this.trainInputs = trainInputs;
            this.trainExpected = trainExpected;
            this.testInputs = testInputs;
            this.testExpected = testExpected;
        }
    }

    public SplitResult split(double[][] inputs, double[][] expected, double trainRatio) {
        return split(inputs, expected, trainRatio, true);
    }

    public SplitResult split(double[][] inputs, double[][] expected, double trainRatio, boolean shuffle) {
        return split(inputs, expected, trainRatio, shuffle, System.currentTimeMillis());
    }

    public SplitResult split(double[][] inputs, double[][] expected, double trainRatio, boolean shuffle, long seed) {
        if (inputs.length != expected.length) {
            throw new IllegalArgumentException("Inputs and expected must have same length");
        }

        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1 (exclusive)");
        }

        int numSamples = inputs.length;
        int trainSize = (int) (numSamples * trainRatio);
        int testSize = numSamples - trainSize;

        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        if (shuffle) {
            shuffleArray(indices, new Random(seed));
        }

        double[][] trainInputs = new double[trainSize][];
        double[][] trainExpected = new double[trainSize][];
        double[][] testInputs = new double[testSize][];
        double[][] testExpected = new double[testSize][];

        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = inputs[indices[i]].clone();
            trainExpected[i] = expected[indices[i]].clone();
        }

        for (int i = 0; i < testSize; i++) {
            testInputs[i] = inputs[indices[trainSize + i]].clone();
            testExpected[i] = expected[indices[trainSize + i]].clone();
        }

        return new SplitResult(trainInputs, trainExpected, testInputs, testExpected);
    }

    private void shuffleArray(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
