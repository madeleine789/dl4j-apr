# Usage guide

This guide demonstrates how to use APR project written in Java with Deeplearning4j.

### Checking out the project
Check out this project from the repository and in console enter to the root directory of the project.

### Corpus preparation
Download and unpack the [PAN-AP-15](https://www.uni-weimar.de/medien/webis/events/pan-15/pan15-web/author-profiling
.html) corpus and [PAN-AP-14](https://www.uni-weimar.de/medien/webis/events/pan-14/pan14-web/author-profiling
.html) corpus using provided instructions.

After unpacking, place the PAN-AP-15 corpus in `src/main/resources/xml` in directories named 
`pan15-[language]`.
To create CSV version of corpus use method `toCSV` from `parsing.pan15.Pan15Parser` class (CSV files will be saved in 
`src/main/resources/supervised/pan15/corpora`).
To divide CSV corpus in training and test sets use method `divideFiles` from `parsing.pan15.Pan15Parser` class.
After set division place both subsets with corresponding `truth.txt` file in directory  
`src/main/resources/supervised/pan15/[language]`.

After unpacking, place the PAN-AP-14 corpus (only twitter folders) in `src/main/resources/xml` in directories named 
`pan14-[language]`. Again, create CSV version (no division) and place files in `src/main/resources/pan14`.

### Model preparation
All parameters are placed in `model.Config` class as final fields.
For models like BagOfWords, Bigram and Word2Vec use constructors from corresponding classes in package `nlp.model` to set `Config.MODEL`.
 
To use Doc2Vec(Tweet2Vec) model you have to create text files containing lines with original tweet and vector 
coordinates separated by comma. They should be placed in directory `src/main/resources/doc2vec/[language]-doc2vec.txt`.
It also needs to be done for pretraining data (files: `src/main/resources/doc2vec/pretr/[language]-doc2vec.txt`).

### Hyperparameter Optimization
Class `nn.dl4j.HyperParamOptimization` contains `main` method responsible for optimization. Prior to running, adjust 
all class fields accordingly.

### Training and Evaluation
Class `Main` is used to train and evaluate DBNs for all available languages. It logs evaluation results after 
training each neural network.