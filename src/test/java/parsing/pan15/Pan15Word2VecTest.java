package parsing.pan15;

import nlp.Pan15SentencePreProcessor;
import nlp.Pan15Word2Vec;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.GloVe;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Assert;
import org.junit.Test;
import model.Author;
import model.Language;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class Pan15Word2VecTest {

    @Test
    public void shouldLoadAndCreateSameWord2Vec() {
        //given
        Pan15Parser parser = new Pan15Parser();
        HashMap<String, Pan15Author> english = parser.parseCSVCorpus().get(Language.ENGLISH);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> englishSentences = english.values().stream().map(Author::getDocuments)
                .collect(Collectors.toList())
                .stream().flatMap(List::stream).collect(Collectors.toList());

        SentenceIterator englishIter = new CollectionSentenceIterator(new Pan15SentencePreProcessor(), englishSentences);
        // when
        Word2Vec englishVec = new Word2Vec.Builder()
                .minWordFrequency(6)
                .iterations(15)
                .layerSize(250)
                .seed(42)
                .windowSize(5)
                .iterate(englishIter)
                .tokenizerFactory(t)
                .build();

        englishVec.fit();
        Word2Vec loadedEnglishVec1 = new Pan15Word2Vec(new SkipGram<>()).readModelFromFile(Language.ENGLISH);
        Word2Vec loadedEnglishVec2 = new Pan15Word2Vec(new CBOW<>()).readModelFromFile(Language.ENGLISH);
        Word2Vec loadedEnglishVec3 = new Pan15Word2Vec(new GloVe<>()).readModelFromFile(Language.ENGLISH);
        loadedEnglishVec1.setTokenizerFactory(t);
        loadedEnglishVec1.setSentenceIter(englishIter);
        loadedEnglishVec2.setTokenizerFactory(t);
        loadedEnglishVec2.setSentenceIter(englishIter);
        loadedEnglishVec3.setTokenizerFactory(t);
        loadedEnglishVec3.setSentenceIter(englishIter);

        //then
        Assert.assertNotNull(loadedEnglishVec1);
        System.out.println(englishVec.wordsNearest("home", 15));
        System.out.println(loadedEnglishVec1.wordsNearest("home", 15));
        System.out.println(loadedEnglishVec2.wordsNearest("home", 15));
        System.out.println(loadedEnglishVec3.wordsNearest("home", 15));
    }
}
