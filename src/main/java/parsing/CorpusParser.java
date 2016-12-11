package parsing;

import model.Author;
import model.Language;

import java.util.HashMap;

public interface CorpusParser<T extends Author> {
    HashMap<Language, HashMap<String, T>> parseXMLCorpus(String path);
    HashMap<Language, HashMap<String, T>> parseCSVCorpus();
}
