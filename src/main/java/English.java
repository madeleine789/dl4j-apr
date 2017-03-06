import model.Config;
import model.Language;
import nn.dl4j.DBN;

public class English {
    public static void main(String... args) {
        DBN dbn = new DBN(
                Language.ENGLISH,
                Config.MODEL,
                Config.PATH + "/english/english-test-pan15.csv",
                Config.PATH + "/english/english-train-pan15.csv"
        );
        dbn.runTrainingAndValidate();
    }
}