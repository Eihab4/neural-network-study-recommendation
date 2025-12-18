package nn.data.handlers;

public class SkipRowHandler implements MissingValueHandler {

    @Override
    public Double handle(int column, double[] columnMeans) {
        return null;
    }
}
