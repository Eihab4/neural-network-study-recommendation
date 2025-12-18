package nn.data.handlers;

public class UseZeroHandler implements MissingValueHandler {

    @Override
    public Double handle(int column, double[] columnMeans) {
        return 0.0;
    }
}
