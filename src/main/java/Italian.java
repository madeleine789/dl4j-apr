import model.Config;
import model.Language;
import nn.dl4j.DBN;

public class Italian {
    public static void main(String... args) throws Exception {
        DBN dbn = new DBN(
                Language.ITALIAN,
                Config.MODEL,
                Config.PATH + "/italian/italian-test-pan15.csv",
                Config.PATH + "/italian/italian-train-pan15.csv"
        );
        dbn.runTrainingAndValidate();
    }
}
