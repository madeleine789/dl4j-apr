import model.Config;
import model.Language;
import nn.dl4j.DBN;

public class Spanish {
    public static void main(String... args) {
        DBN dbn = new DBN(
                Language.SPANISH,
                Config.MODEL,
                Config.PATH + "/spanish/spanish-test-pan15.csv",
                Config.PATH + "/spanish/spanish-train-pan15.csv"
        );
        dbn.runTrainingAndValidate();
    }
}
