import nn.dl4j.DBN;
import model.Language;

public class Spanish {
    public static void main(String... args) {
        DBN dbn = new DBN(
                Language.SPANISH,
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/spanish/spanish-test" +
                        "-pan15.csv",
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/spanish/spanish-train" +
                        "-pan15.csv"
        );
        dbn.runTrainingAndValidate();
    }
}
