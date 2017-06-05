import model.Config;
import model.Language;
import nn.dl4j.DBN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String... args) throws Exception {
        DBN dbn;

        log.info("==============================DUTCH==============================");

        dbn = new DBN(
                Language.DUTCH,
                Config.PATH + "/dutch/dutch-test-pan15.csv",
                Config.PATH + "/dutch/dutch-train-pan15.csv",
                false);

        dbn.trainWithEarlyStopping();

        log.info("==============================ITALIAN==============================");

        dbn = new DBN(
                Language.ITALIAN,
                Config.PATH + "/italian/italian-test-pan15.csv",
                Config.PATH + "/italian/italian-train-pan15.csv",
                false);

        dbn.trainWithEarlyStopping();

        log.info("==============================ENGLISH==============================");

        dbn = new DBN(
                Language.ENGLISH,
                Config.PATH + "/english/english-test-pan15.csv",
                Config.PATH + "/english/english-train-pan15.csv",
                false);
        dbn.trainWithEarlyStopping();

        log.info("==============================SPANISH==============================");

        dbn = new DBN(
                Language.SPANISH,
                Config.PATH + "/spanish/spanish-test-pan15.csv",
                Config.PATH + "/spanish/spanish-train-pan15.csv",
                false);
        dbn.trainWithEarlyStopping();
    }


}
