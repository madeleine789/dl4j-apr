package nlp;

import model.Language;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Model {
    int getVecLength();
    INDArray getVector(String sentence, Language language);
}
