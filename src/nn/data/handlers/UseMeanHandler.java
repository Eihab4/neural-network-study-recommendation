package nn.data.handlers;

public class UseMeanHandler implements MissingValueHandler {

    @Override
    public Double handle(int column, double[] columnMeans) {
        if (columnMeans == null || column >= columnMeans.length) {
            return 0.0;
        }
        return columnMeans[column];
    }
}
