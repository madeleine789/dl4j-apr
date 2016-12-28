package parsing.pan15;

import parsing.CorpusParser;
import model.Language;
import model.Personality;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class Pan15Parser implements CorpusParser<Pan15Author> {

    static HashMap<Language, List<String[]>> truthFiles;

    static {
        try {
            truthFiles = getTruthFiles();
        } catch (URISyntaxException | IOException e) {
            e.printStackTrace();
        }
    }

    private static Pan15Author parseDocument(File file) throws IOException, JAXBException {

        JAXBContext jaxbContext = JAXBContext.newInstance(Pan15Author.class);
        Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
        return (Pan15Author) jaxbUnmarshaller.unmarshal(file);
    }

     static HashMap<String, Pan15Author> parseLanguage(Language language) {
        HashMap<String, Pan15Author> authors = new HashMap<>();
        File path = new File("/Users/mms/Desktop/PR_DNN/dl4j-apr/src/main/resources/supervised/pan15/corpora/" +
                language.getName() + ".csv");
        try {
            Files.readAllLines(path.toPath()).stream().filter(l -> ! l.startsWith("\"AUTHID\"")).forEach(l -> {
                String[] fields = l.split(",");
                String id = fields[0].substring(1, fields[0].lastIndexOf("\""));
                String doc = fields[1].substring(1, fields[1].lastIndexOf("\""));
                Pan15Author a;
                if (authors.containsKey(id)) {
                    a = authors.get(id);
                    a.addDocument(doc);
                } else {
                    a = new Pan15Author();
                    a.setId(id);
                    a.setLang(language.getName());
                    a.addDocument(doc);
                    fillAuthorFields(a, language);
                    authors.put(id, a);
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return authors;
    }

    public static HashMap<String, Pan15Author> parseLanguage(String path, Language language) {
        URL resource = Pan15Parser.class.getClassLoader().getResource(path);
        File folder = new File(resource.getPath());
        File[] files = folder.listFiles();
        List<File> xmlFiles = Arrays.stream(files).filter(f -> f.getName().endsWith(".xml")).collect(Collectors.toList());
        HashMap<String, Pan15Author> authors = new HashMap<>();
        xmlFiles.forEach(f -> {
            Pan15Author a = null;
            try {
                a = fillAuthorFields(parseDocument(f), language);
            } catch (IOException | JAXBException e) {
                e.printStackTrace();
            } finally {
                if (a != null) authors.put(a.getId(), a);
            }
        });
        return authors;
    }

    public HashMap<Language, HashMap<String, Pan15Author>> parseCSVCorpus() {
        URL resource = Pan15Parser.class.getClassLoader().getResource("supervised/pan15/corpora");
        File folder = new File(resource.getPath());
        File[] files = folder.listFiles();
        HashMap<Language, HashMap<String, Pan15Author>> result = new HashMap<>();
        List<File> csvFiles = Arrays.stream(files).filter(f -> f.getName().endsWith(".csv")).collect(Collectors.toList());

        csvFiles.forEach(f -> {
            Language language = Language.valueOf(f.getName().substring(0, f.getName().lastIndexOf('.')).toUpperCase());
            HashMap<String, Pan15Author> authors = new HashMap<>();
            try {
                Files.readAllLines(f.toPath()).stream().filter(l -> ! l.startsWith("\"AUTHID\"")).forEach(l -> {
                    String[] fields = l.split(",");
                    String id = fields[0].substring(1, fields[0].lastIndexOf("\""));
                    String doc = fields[1].substring(1, fields[1].lastIndexOf("\""));
                    Pan15Author a;
                    if (authors.containsKey(id)) {
                        a = authors.get(id);
                        a.addDocument(doc);
                    } else {
                        a = new Pan15Author();
                        a.setId(id);
                        a.setLang(language.getName());
                        a.addDocument(doc);
                        fillAuthorFields(a, language);
                        authors.put(id, a);
                    }
                });
            } catch (IOException e) {
                e.printStackTrace();
            }
            result.put(language,authors);
        });
        return result;
    }

    private static Pan15Author fillAuthorFields(Pan15Author author, Language language) {
        String[] truth = truthFiles.get(language).stream().filter(s -> Objects.equals(s[0], author.getId())).findFirst().get();
        author.setGender(truth[1]);
        author.setAge(truth[2]);
        author.setPersonality(Personality.E, Double.parseDouble(truth[3]));
        author.setPersonality(Personality.N, -1 * Double.parseDouble(truth[4])); // this is stable - opposite
        author.setPersonality(Personality.A, Double.parseDouble(truth[5]));
        author.setPersonality(Personality.C, Double.parseDouble(truth[6]));
        author.setPersonality(Personality.O, Double.parseDouble(truth[7]));
        return author;
    }

    private static HashMap<Language,List<String[]>> getTruthFiles() throws URISyntaxException, IOException {
        HashMap<Language, List<String[]>> result = new HashMap<>();
        for(Language language : Language.values()) {
            URL resource = Pan15Parser.class.getClassLoader().getResource("supervised/pan15/" + language.getName() +
                    "/truth.txt");
            List<String[]> lines = Files.readAllLines(new File(resource.getPath()).toPath()).stream().map(l -> l
                    .split(":::")).collect(Collectors.toList());
            result.put(language, lines);
        }
        return result;
    }

    @Override
    public HashMap<Language, HashMap<String, Pan15Author>> parseXMLCorpus(String path) {
        HashMap<Language, HashMap<String, Pan15Author>> result = new HashMap<>();
        for(Language l : Language.values()) {
            result.put(l, parseLanguage(path, l));
        }
        return result;
    }

    public static void main(String... args) throws IOException, JAXBException, URISyntaxException {
        System.out.println(new Pan15Parser().parseCSVCorpus().get(Language.ENGLISH).size());
    }
}
