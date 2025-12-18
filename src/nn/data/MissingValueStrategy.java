package nn.data;

public enum MissingValueStrategy {
    THROW_ERROR,
    USE_ZERO,
    USE_MEAN,
    SKIP_ROW
}
