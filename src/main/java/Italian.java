import model.Language;
import nn.dl4j.DBN;

public class Italian {
    public static void main(String... args) throws Exception {
        DBN dbn = new DBN(
                Language.ITALIAN,
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/italian/italian-test" +
                        "-pan15.csv",
                "/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/italian/italian-train" +
                        "-pan15.csv"
        );
        dbn.runTrainingAndValidate();
    }
}
