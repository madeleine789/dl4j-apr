package parsing.pan15;

import model.Language;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;


public class Pan15ParserTest {
    @Test
    public void shouldParseEnglishLanguage() {
        //given
        Language l = Language.ENGLISH;
        Pan15Parser parser = new Pan15Parser();
        //when
        HashMap<Language, HashMap<String, Pan15Author>> pan15Corpus = parser.parseCSVCorpus();
        //then
        Assert.assertEquals(pan15Corpus.get(l).keySet().size(), 152);
    }
}
