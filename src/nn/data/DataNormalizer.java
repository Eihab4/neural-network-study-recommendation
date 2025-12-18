package nn.data;

public class DataNormalizer {

    private double[] inputMins;
    private double[] inputMaxs;
    private double[] outputMins;
    private double[] outputMaxs;
    private boolean fitted = false;

    public void fit(double[][] inputs, double[][] outputs) {
        if (inputs.length == 0 || outputs.length == 0) {
            throw new IllegalArgumentException("Cannot fit on empty data");
        }

        int inputFeatures = inputs[0].length;
        int outputFeatures = outputs[0].length;

        inputMins = new double[inputFeatures];
        inputMaxs = new double[inputFeatures];
        outputMins = new double[outputFeatures];
        outputMaxs = new double[outputFeatures];

        for (int j = 0; j < inputFeatures; j++) {
            inputMins[j] = Double.MAX_VALUE;
            inputMaxs[j] = Double.MIN_VALUE;
        }

        for (int j = 0; j < outputFeatures; j++) {
            outputMins[j] = Double.MAX_VALUE;
            outputMaxs[j] = Double.MIN_VALUE;
        }

        for (double[] input : inputs) {
            for (int j = 0; j < inputFeatures; j++) {
                if (input[j] < inputMins[j])
                    inputMins[j] = input[j];
                if (input[j] > inputMaxs[j])
                    inputMaxs[j] = input[j];
            }
        }

        for (double[] output : outputs) {
            for (int j = 0; j < outputFeatures; j++) {
                if (output[j] < outputMins[j])
                    outputMins[j] = output[j];
                if (output[j] > outputMaxs[j])
                    outputMaxs[j] = output[j];
            }
        }

        fitted = true;
    }

    public double[][] normalizeInputs(double[][] inputs) {
        checkFitted();
        double[][] result = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = normalizeInput(inputs[i]);
        }
        return result;
    }

    public double[] normalizeInput(double[] input) {
        checkFitted();
        double[] result = new double[input.length];
        for (int j = 0; j < input.length; j++) {
            double range = inputMaxs[j] - inputMins[j];
            if (range == 0) {
                result[j] = 0;
            } else {
                result[j] = (input[j] - inputMins[j]) / range;
            }
        }
        return result;
    }

    public double[][] normalizeOutputs(double[][] outputs) {
        checkFitted();
        double[][] result = new double[outputs.length][];
        for (int i = 0; i < outputs.length; i++) {
            result[i] = normalizeOutput(outputs[i]);
        }
        return result;
    }

    public double[] normalizeOutput(double[] output) {
        checkFitted();
        double[] result = new double[output.length];
        for (int j = 0; j < output.length; j++) {
            double range = outputMaxs[j] - outputMins[j];
            if (range == 0) {
                result[j] = 0;
            } else {
                result[j] = (output[j] - outputMins[j]) / range;
            }
        }
        return result;
    }

    public double[] denormalizeInput(double[] normalized) {
        checkFitted();
        double[] result = new double[normalized.length];
        for (int j = 0; j < normalized.length; j++) {
            result[j] = normalized[j] * (inputMaxs[j] - inputMins[j]) + inputMins[j];
        }
        return result;
    }

    public double[] denormalizeOutput(double[] normalized) {
        checkFitted();
        double[] result = new double[normalized.length];
        for (int j = 0; j < normalized.length; j++) {
            result[j] = normalized[j] * (outputMaxs[j] - outputMins[j]) + outputMins[j];
        }
        return result;
    }

    public double denormalizeOutputValue(double normalized, int index) {
        checkFitted();
        return normalized * (outputMaxs[index] - outputMins[index]) + outputMins[index];
    }

    private void checkFitted() {
        if (!fitted) {
            throw new IllegalStateException("DataNormalizer must be fitted before use");
        }
    }

    public double[] getInputMins() {
        return inputMins != null ? inputMins.clone() : null;
    }

    public double[] getInputMaxs() {
        return inputMaxs != null ? inputMaxs.clone() : null;
    }

    public double[] getOutputMins() {
        return outputMins != null ? outputMins.clone() : null;
    }

    public double[] getOutputMaxs() {
        return outputMaxs != null ? outputMaxs.clone() : null;
    }
}
