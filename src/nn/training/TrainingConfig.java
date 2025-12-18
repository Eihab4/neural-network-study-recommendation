package nn.training;

public class TrainingConfig {

    private double learningRate;
    private int epochs;
    private boolean shuffle;
    private boolean verbose;
    private int printEvery;

    public TrainingConfig() {
        this.learningRate = 0.01;
        this.epochs = 1000;
        this.shuffle = true;
        this.verbose = true;
        this.printEvery = 100;
    }

    public TrainingConfig setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public TrainingConfig setEpochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public TrainingConfig setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    public TrainingConfig setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public TrainingConfig setPrintEvery(int printEvery) {
        this.printEvery = printEvery;
        return this;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public boolean isShuffle() {
        return shuffle;
    }

    public boolean isVerbose() {
        return verbose;
    }

    public int getPrintEvery() {
        return printEvery;
    }
}
