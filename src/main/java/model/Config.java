package model;

import nlp.model.*;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public final class Config {
    public static final String PATH = "./src/main/resources/supervised/pan15";
    public static final String RESOURCE_PATH = "supervised/pan15/";
    public static final Model MODEL = new Pan15Doc2Vec();//new Pan15BagOfWords(false);
    public static final LossFunctions.LossFunction LOSS_FUNCTION = LossFunctions.LossFunction.MSE;
    public static final Updater UPDATER = Updater.ADADELTA;
    public static final double DROPOUT = 0.5;
    public static final int BATCH_SIZE = 500;
    public static final int NUM_OUTPUTS = 5;
    public static final int IDX_FROM = 2; // index of column with first OCEAN trait in csv file
    public static final int IDX_TO = 6; // index of column with last trait in csv file
    public static final boolean REGRESSION = true; // index of column with last trait in csv file
}
