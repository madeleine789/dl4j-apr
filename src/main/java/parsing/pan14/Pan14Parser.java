package parsing.pan14;

import model.Author;
import model.Language;
import org.jsoup.Jsoup;
import parsing.CorpusParser;
import parsing.pan15.Pan15Parser;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.annotation.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class Pan14Parser implements CorpusParser<Pan14Parser.Pan14Author> {

    private static Pan14Author parseDocument(File file) throws IOException, JAXBException {

        JAXBContext jaxbContext = JAXBContext.newInstance(Pan14Author.class);
        Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
        return (Pan14Author) jaxbUnmarshaller.unmarshal(file);
    }

    public static HashMap<String, Pan14Author> parseLanguage(String path, Language language) throws IOException, URISyntaxException {
        HashMap<String, String[]> truth = getTruthFile(language);
        path = path + language.getName() + "/";
        URL resource = Pan15Parser.class.getClassLoader().getResource(path);
        File folder = new File(resource.getPath());
        File[] files = folder.listFiles();
        List<File> xmlFiles = Arrays.stream(files).filter(f -> f.getName().endsWith(".xml")).collect(Collectors.toList());
        HashMap<String, Pan14Author> authors = new HashMap<>();
        xmlFiles.forEach(f -> {
            Pan14Author a = null;
            try {
                a = parseDocument(f);

                a.setId(f.getName().substring(0, f.getName().indexOf(".xml")));

                String[] t = truth.get(a.getId());
                a.setGender(String.valueOf(t[0].charAt(0)));
                a.setAge(t[1]);
                a.setDocuments(a.getDocuments().stream().map(d -> Jsoup.parse(d).text()).collect(Collectors.toList()));
            } catch (IOException | JAXBException e) {
                e.printStackTrace();
            } finally {
                if (a != null) authors.put(a.getId(), a);
            }
        });
        return authors;
    }



    private static HashMap<String, String[]> getTruthFile(Language language) throws URISyntaxException, IOException {
        HashMap<String, String[]> truth = new HashMap<>();
        URL resource = Pan15Parser.class.getClassLoader().getResource("pan14/twitter-" + language.getName() +
                "/truth.txt");
        List<String> lines =  Files.readAllLines(new File(resource.getPath()).toPath());
        for(String l: lines) {
            String[] items = l.split(":::");
            truth.put(items[0], new String[]{items[1], items[2]});
        }
        return truth;
    }


    @Override
    public HashMap<Language, HashMap<String, Pan14Author>> parseXMLCorpus(String path) {
        HashMap<Language, HashMap<String, Pan14Author>> result = new HashMap<>();

        try {
            result.put(Language.ENGLISH, parseLanguage(path, Language.ENGLISH));
            result.put(Language.SPANISH, parseLanguage(path, Language.SPANISH));
        } catch (IOException | URISyntaxException e) {
            e.printStackTrace();
        }

        return result;
    }

    private void toCSV(String path) throws FileNotFoundException {
        HashMap<Language, HashMap<String, Pan14Author>> map = parseXMLCorpus("pan14/twitter-");
        for (Language l: map.keySet()) {
            HashMap<String, Pan14Author> result = map.get(l);
            PrintWriter pw = new PrintWriter(new File("" + l.getName() + ".csv"));
            pw.write("\"AUTHID\",\"STATUS\",\"GENDER\",\"AGE\"\n");
            for (String id : result.keySet()) {
                Pan14Author author = result.get(id);
                for (String doc : author.getDocuments()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append('"').append(author.getId()).append('"').append(',');
                    sb.append('"').append(doc.trim().replaceAll("\n", " ").replaceAll(",", "")).append('"').append(',');
                    sb.append(author.getGender()).append(',');
                    sb.append(author.getAge()).append("\n");
                    pw.write(sb.toString());
                }
            }
            pw.close();
        }
    }

    @Override
    public HashMap<Language, HashMap<String, Pan14Author>> parseCSVCorpus() {
        List<Language> langs = Arrays.asList(Language.ENGLISH, Language.SPANISH);
        HashMap<Language, HashMap<String, Pan14Author>> result = new HashMap<>();
        for (Language lang : langs) {
            URL resource = Pan15Parser.class.getClassLoader().getResource("pan14/" + lang.getName() + ".csv");
            try {
                HashMap<String, Pan14Author> authors = new HashMap<>();
                List<String> lines =  Files.readAllLines(new File(resource.getPath()).toPath());
                for (String l : lines) {
                    if (l.startsWith("\"AUTHID\"")) continue;
                    String[] fields = l.split(",");
                    String id = fields[0].substring(1, fields[0].lastIndexOf("\""));
                    String doc = fields[1].substring(1, fields[1].lastIndexOf("\""));
                    Pan14Author a;
                    if (authors.containsKey(id)) {
                        a = authors.get(id);
                        a.addDocument(doc);
                    } else {
                        a = new Pan14Author();
                        a.setId(id);
                        a.setLang(lang.getName());
                        a.addDocument(doc);
                        a.setAge(fields[3]);
                        a.setGender(fields[2]);
                        authors.put(id, a);
                    }

                }
                result.put(lang, authors);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    @XmlRootElement(name = "author")
    @XmlAccessorType(XmlAccessType.FIELD)
    static class Pan14Author implements Author {
        private String id;
        @XmlAttribute
        private String lang;
        @XmlElementWrapper(name = "documents")
        @XmlElement(name = "document")
        private List<String> documents;
        private String gender;
        private String age;

        public String getId() {
            return id;
        }

        public String getLanguage() {
            return lang;
        }

        public List<String> getDocuments() {
            return documents;
        }

        public void setId(String id) {
            this.id = id;
        }

        public void setLang(String lang) {
            this.lang = lang;
        }

        public void setDocuments(List<String> documents) {
            this.documents = documents;
        }

        public void addDocument(String doc) {
            if (documents!= null && !documents.isEmpty()) documents.add(doc);
            else {
                documents = new ArrayList<>();
                documents.add(doc);
            }
        }

        public void setGender(String gender) {
            this.gender = gender;
        }

        public void setAge(String age) {
            this.age = age;
        }

        public String getAge() {
            return age;
        }

        public String getGender() {
            return gender;
        }
    }

    public static void main(String... args) {
        Pan14Parser p = new Pan14Parser();
        HashMap<Language, HashMap<String, Pan14Author>> map = p.parseCSVCorpus();
        System.out.println(map.keySet().size());
        System.out.println(map.get(Language.SPANISH).keySet().size());
    }
}
