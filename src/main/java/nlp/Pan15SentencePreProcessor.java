package nlp;

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.text.Normalizer;
import java.util.regex.Pattern;

public class Pan15SentencePreProcessor implements SentencePreProcessor {

    private boolean deleteNonASCIIcharacters;

    private String doc;
    public final String P_HYPERLINK = " P_HYPERLINK ";
    public final String P_CURRENCY = " P_CURRENCY ";
    private static final String P_EMOTICON = " P_EMOTICON ";
    private static final String P_DATE = "P_DATE";
    private static final String P_NUMBER = " P_NUMBER ";
    private static final String P_QUOTATION = "P_QUOTATION";

    public Pan15SentencePreProcessor(boolean deleteNonASCIIcharacters) {
        this.deleteNonASCIIcharacters = deleteNonASCIIcharacters;
    }

    public Pan15SentencePreProcessor() {
        this.deleteNonASCIIcharacters = true;
    }

    public String preProcess(String doc) {
        this.doc = doc;
        cleanLinks();
        cleanEmoticons();
        cleanAmpersands();
        cleanCurrency();
        cleanDates();
        cleanNumbers();
        cleanWeirdQuotationMarks();
//        cleanQuotes();
        squeezWhiteSpaces();

        if (deleteNonASCIIcharacters){
            this.doc = Normalizer.normalize(this.doc, Normalizer.Form.NFD);
            this.doc = this.doc.replaceAll("[^\\x00-\\x7F]", "");
        }

        return this.doc;
    }

    private void cleanLinks() {
        String regex = "https?://[^(),;'\"“”\\s]+";
        doc = doc.replaceAll(regex, P_HYPERLINK);
    }

    private void cleanCurrency() {
        String regex = "((\\$|€|¥|£)([0-9]+[Mm]?)((\\.|,)([0-9]*))?)|(([0-9]+M?)((\\.|,)([0-9]*))?\\s?(\\$|€|¥|£))";
        doc = doc.replaceAll(regex, P_CURRENCY);
    }

    private void cleanEmoticons() {
        String[] patterns = new String[]{
                "<+3+",
                "\\s((?::|;|=|8)(?:-)?(?:')?(?:'-)?(?:\\)|(?:\\()|(?:\\[)|[dD]|[pP]|[oO]|[|/*3v$]))(?=\\s)",
                "(\\s[xX][dD]+)(?=[\\s|\\p{Punct}\\s])",
                "(\\s[xX][oO]+)(?=[\\s|\\p{Punct}\\s])",
                "(\\s[xX][pP]+)(?=[\\s|\\p{Punct}\\s])",
                "(\\s[T*+ó0OoxX]_[T*+ò0OoxX])(?=[\\s|\\p{Punct}\\s])"
        };
        for (String pattern: patterns) {
            doc = Pattern.compile(pattern).matcher(doc).replaceAll(P_EMOTICON).trim();
        }
    }

    private void cleanDates() {
        String[] patterns = new String[] {
                "\\b(?<![.-/])(0?[1-9]|[1][1-2])[-./](0[1-9]|[12][0-9]|3[01])[-./](18|19|20|21)\\d{2}\\b(?![.-/])",
                "\\b(?<![.-/])(0?[1-9]|[12][0-9]|3[01])[-./](0?[1-9]|[1][1-2])[-./](18|19|20|21)\\d{2}\\b(?![.-/])",
                "\\b(?<![.-/])(18|19|20|21)\\d{2}[-./](0?[1-9]|[1][1-2])[-./](0?[1-9]|[12][0-9]|3[01])\\b(?![.-/])",
                "\\b(?<![.-/])(18|19|20|21)\\d{2}[-./](0?[1-9]|[12][0-9]|3[01])[-./](0?[1-9]|[1][1-2])\\b(?![.-/])",
                "\\b(?<![-/])(0[1-9]|[1][1-2])[/-](18|19|20|21)\\d{2}\\b(?![-/])",
                "\\b(?<![-/])(18|19|20|21)\\d{2}[/-](0[1-9]|[1][1-2])\\b(?![-/])",
                "(18|19|20|21)\\d{2}"//([-/](18|19|20|21)\\d{2})?" // year to clear or not to clear

        };
        for (String pattern: patterns) {
            doc = Pattern.compile(pattern).matcher(doc).replaceAll(P_DATE).trim();
        }

    }

    private void cleanNumbers() {
        String[] patterns = new String[] {
                "(?<=[\\p{Punct}\\s])[0-9]+([.,]?[0-9]+)?(?=[\\s|\\p{Punct}\\s])",
                "\\p{Sc}?\\d+(?:\\.\\d+)+%?"
        };

        for (String pattern: patterns) {
            doc = Pattern.compile(pattern).matcher(doc).replaceAll(P_NUMBER).trim();
        }

    }

    private void cleanWeirdQuotationMarks() {
        doc = doc.replaceAll("(”|“)", "\"");
    }

    private void cleanQuotes() {
        doc = doc.replaceAll("\"[^\"]{100,}\"", P_QUOTATION);
    }

    private void squeezWhiteSpaces() {
        doc = Pattern.compile("[\\s\\p{Zs}]+").matcher(doc).replaceAll(" ").trim();
    }

    public void cleanAmpersands() {
        doc = doc.replaceAll("&amp;", "and");
        doc = doc.replaceAll("&", "and");
    }
}
