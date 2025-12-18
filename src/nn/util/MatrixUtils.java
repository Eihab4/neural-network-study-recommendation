package nn.util;

public class MatrixUtils {

    public static double[][] multiply(double[][] a, double[][] b) {
        validateNotNull(a, "First matrix");
        validateNotNull(b, "Second matrix");
        validateNotEmpty(a, "First matrix");
        validateNotEmpty(b, "Second matrix");

        int aRows = a.length;
        int aCols = a[0].length;
        int bRows = b.length;
        int bCols = b[0].length;

        if (aCols != bRows) {
            throw new IllegalArgumentException(
                    "Cannot multiply matrices: A columns (" + aCols +
                            ") != B rows (" + bRows + ")");
        }

        double[][] result = new double[aRows][bCols];

        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0.0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
        validateSameShape(a, b, "add");

        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    public static double[][] subtract(double[][] a, double[][] b) {
        validateSameShape(a, b, "subtract");

        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }

        return result;
    }

    public static double[][] multiplyElementWise(double[][] a, double[][] b) {
        validateSameShape(a, b, "multiply element-wise");

        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }

        return result;
    }

    public static double[][] scale(double[][] matrix, double scalar) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }

        return result;
    }

    public static double[][] addScalar(double[][] matrix, double scalar) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] + scalar;
            }
        }

        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    public static double[][] toRowMatrix(double[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }

        double[][] result = new double[1][vector.length];
        result[0] = vector.clone();
        return result;
    }

    public static double[][] toColumnMatrix(double[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }

        double[][] result = new double[vector.length][1];
        for (int i = 0; i < vector.length; i++) {
            result[i][0] = vector[i];
        }
        return result;
    }

    public static double[] getRow(double[][] matrix, int row) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        if (row < 0 || row >= matrix.length) {
            throw new IllegalArgumentException(
                    "Row index " + row + " out of bounds for matrix with " +
                            matrix.length + " rows");
        }

        return matrix[row].clone();
    }

    public static double[] flatten(double[][] rowMatrix) {
        validateNotNull(rowMatrix, "Matrix");
        validateNotEmpty(rowMatrix, "Matrix");

        if (rowMatrix.length != 1) {
            throw new IllegalArgumentException(
                    "Can only flatten row matrix [1 × n], got [" +
                            rowMatrix.length + " × " + rowMatrix[0].length + "]");
        }

        return rowMatrix[0].clone();
    }

    public static double[][] zeros(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException(
                    "Matrix dimensions must be positive: [" + rows + " × " + cols + "]");
        }
        return new double[rows][cols];
    }

    public static double[][] copy(double[][] matrix) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            result[i] = matrix[i].clone();
        }

        return result;
    }

    public static int[] shape(double[][] matrix) {
        validateNotNull(matrix, "Matrix");
        validateNotEmpty(matrix, "Matrix");

        return new int[] { matrix.length, matrix[0].length };
    }

    public static String toString(double[][] matrix) {
        if (matrix == null)
            return "null";
        if (matrix.length == 0)
            return "[]";

        StringBuilder sb = new StringBuilder();
        sb.append("Matrix [").append(matrix.length).append(" × ").append(matrix[0].length).append("]\n");

        for (double[] row : matrix) {
            sb.append("  [");
            for (int j = 0; j < row.length; j++) {
                sb.append(String.format("%8.4f", row[j]));
                if (j < row.length - 1)
                    sb.append(", ");
            }
            sb.append("]\n");
        }

        return sb.toString();
    }

    private static void validateNotNull(double[][] matrix, String name) {
        if (matrix == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
    }

    private static void validateNotEmpty(double[][] matrix, String name) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
    }

    private static void validateSameShape(double[][] a, double[][] b, String operation) {
        validateNotNull(a, "First matrix");
        validateNotNull(b, "Second matrix");
        validateNotEmpty(a, "First matrix");
        validateNotEmpty(b, "Second matrix");

        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException(
                    "Cannot " + operation + " matrices with different shapes: [" +
                            a.length + " × " + a[0].length + "] vs [" +
                            b.length + " × " + b[0].length + "]");
        }
    }
}
