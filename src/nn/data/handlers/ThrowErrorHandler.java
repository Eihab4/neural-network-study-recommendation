package nn.data.handlers;

public class ThrowErrorHandler implements MissingValueHandler {

    @Override
    public Double handle(int column, double[] columnMeans) {
        throw new RuntimeException("Missing or invalid value at column " + column);
    }
}
