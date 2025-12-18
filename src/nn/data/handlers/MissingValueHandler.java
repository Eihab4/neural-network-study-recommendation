package nn.data.handlers;

public interface MissingValueHandler {

    Double handle(int column, double[] columnMeans);
}
