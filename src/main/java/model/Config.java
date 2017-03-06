package model;

import nlp.Model;
import nlp.Pan15BagOfWords;

public final class Config {
    public static final String PATH = "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15";
    public static final Model MODEL = new Pan15BagOfWords(true);
}
