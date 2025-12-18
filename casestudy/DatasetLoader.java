package casestudy;

import nn.data.handlers.MissingValueHandler;
import nn.data.handlers.ThrowErrorHandler;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DatasetLoader {

    private MissingValueHandler handler;
    private double[][] inputs;
    private double[][] expected;
    private double[] columnMeans;
    private int inputColumns;
    private int outputColumns;

    public DatasetLoader() {
        this(new ThrowErrorHandler());
    }

    public DatasetLoader(MissingValueHandler handler) {
        this.handler = handler;
    }

    public void load(String filename, int numInputColumns, int numOutputColumns) throws IOException {
        this.inputColumns = numInputColumns;
        this.outputColumns = numOutputColumns;

        List<double[]> inputList = new ArrayList<>();
        List<double[]> expectedList = new ArrayList<>();

        List<String[]> rawRows = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String headerLine = reader.readLine();

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;
                String[] parts = line.split(",");
                rawRows.add(parts);
            }
        }

        calculateColumnMeans(rawRows);

        for (String[] parts : rawRows) {
            boolean skipRow = false;
            double[] inputRow = new double[inputColumns];
            double[] expectedRow = new double[outputColumns];

            for (int i = 0; i < inputColumns; i++) {
                String value = (i < parts.length) ? parts[i].trim() : "";
                Double parsed = parseValue(value, i);
                if (parsed == null) {
                    skipRow = true;
                    break;
                }
                inputRow[i] = parsed;
            }

            if (skipRow)
                continue;

            for (int i = 0; i < outputColumns; i++) {
                int colIndex = inputColumns + i;
                String value = (colIndex < parts.length) ? parts[colIndex].trim() : "";
                Double parsed = parseValue(value, colIndex);
                if (parsed == null) {
                    skipRow = true;
                    break;
                }
                expectedRow[i] = parsed;
            }

            if (skipRow)
                continue;

            inputList.add(inputRow);
            expectedList.add(expectedRow);
        }

        inputs = inputList.toArray(new double[0][]);
        expected = expectedList.toArray(new double[0][]);
    }

    private void calculateColumnMeans(List<String[]> rawRows) {
        int totalColumns = inputColumns + outputColumns;
        columnMeans = new double[totalColumns];
        int[] counts = new int[totalColumns];

        for (String[] row : rawRows) {
            for (int i = 0; i < totalColumns && i < row.length; i++) {
                String value = row[i].trim();
                if (!value.isEmpty()) {
                    try {
                        columnMeans[i] += Double.parseDouble(value);
                        counts[i]++;
                    } catch (NumberFormatException e) {
                    }
                }
            }
        }

        for (int i = 0; i < totalColumns; i++) {
            if (counts[i] > 0) {
                columnMeans[i] /= counts[i];
            }
        }
    }

    private Double parseValue(String value, int column) {
        if (value == null || value.isEmpty()) {
            return handler.handle(column, columnMeans);
        }

        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return handler.handle(column, columnMeans);
        }
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getExpected() {
        return expected;
    }

    public int getSampleCount() {
        return inputs != null ? inputs.length : 0;
    }
}
