import nlp.Pan15BagOfWords;
import nn.dl4j.DBN;
import model.Language;

public class Dutch {
    public static void main(String... args) {
        DBN dbn = new DBN(
                Language.DUTCH,
                new Pan15BagOfWords(),
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/dutch/dutch-test-pan15.csv",
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/dutch/dutch-train-pan15.csv"
        );

        dbn.runTrainingAndValidate();
    }
}
