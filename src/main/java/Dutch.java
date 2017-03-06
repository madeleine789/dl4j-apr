import model.Config;
import model.Language;
import nn.dl4j.DBN;

public class Dutch {
    public static void main(String... args) {
        DBN dbn = new DBN(
                Language.DUTCH,
                Config.MODEL,
                Config.PATH + "/dutch/dutch-test-pan15.csv",
                Config.PATH + "/dutch/dutch-train-pan15.csv"
        );

        dbn.runTrainingAndValidate();
    }
}
