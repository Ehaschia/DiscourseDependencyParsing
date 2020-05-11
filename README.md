# Discourse Dependency Parsing Using Arc-factored Model

- Task: Discourse dependency parsing based on Rhetorical Structure Theory
    - Input: EDUs, syntactic features, template features, sentence/paragraph boundaries
    - Output: RST-style dependency graph

## Requirements ##

- numpy
- spacy >= 2.1.9
- chainer >= 6.1.0
- jsonlines
- pyprind

## Configuration ##

1. Clone this repository and create directories to store preprocessed data and outputs

```
$ git clone https://github.com/norikinishida/DiscourseDependencyParsing-ArcFactoredModel
$ cd ./DiscourseDependencyParsing-ArcFactoredModel
$ mkdir ./data
$ mkdir ./results
```

2. Edit ```./run_preprocessing.sh``` as follows:

```shell
STORAGE=./data
```

3. Edit ```./config/path.ini``` as follows:

```INI
data = "./data"
results = "./results"
pretrained_word_embeddings = "/path/to/your/pretrained_word_embeddings"
rstdt = "/path/to/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0"
```

4. Clone other libraries

```
$ mkdir ./tmp
$ cd ./tmp
$ pip install pandas
$ pip install scikit-learn
$ pip install gensim
$ pip install nltk
$ git clone https://github.com/norikinishida/utils.git
$ git clone https://github.com/norikinishida/treetk.git
$ cp -r ./utils/utils ..
$ cp -r ./treetk/treetk ..
```

## Preprocessing ##

- Run the following script:

```
./run_preprocessing.sh
```

- The following directories will be generated:
    - ./data/rstdt/wsj/{train,test} (preprocessed RST-DT)
    - ./data/ptbwsj_wo_rstdt {preprocessed PTB-WSJ}
    - ./data/rstdt-vocab (vocabularies)

## Parsing Model: Arc-Factored Model ##

- EDU-level feature extraction
    - word embeddings of the beginning/end words
    - POS embeddings of the beginning/end words
    - word/POS/dependency embeddings of the head word
    - template features

- Arc-level feature extraction
    - bidirectional LSTM
    - template features

- Attachment scoring
    - MLP

- Decoding algorithm (unlabeled tree-building)
    - Eisner

- Labeling
    - MLP

## Training ##

- Loss function: Margin-based criterion and softmax cross entropy
- Training data: RST-DT training set
- Run the following command:

```
python main.py --gpu 0 --model arcfactoredmodel --config ./config/hyperparams_1.ini --name trial1 --actiontype train --max_epoch 40
```

- The following files will be generated:
    - ./results/arcfactoredmodel.hyperparams_1.trial1.training.log
    - ./results/arcfactoredmodel.hyperparams_1.trial1.training.jsonl
    - ./results/arcfactoredmodel.hyperparams_1.trial1.model
    - ./results/arcfactoredmodel.hyperparams_1.trial1.valid_pred.arcs (optional)
    - ./results/arcfactoredmodel.hyperparams_1.trial1.valid_gold.arcs (optional)
    - ./results/arcfactoredmodel.hyperparams_1.trial1.validation.jsonl (optional)

## Evaluation ##

- Metrics: Labeled and unlabeled attachment scores (LAS, UAS)
- Test data: RST-DT test set
- Run the following command:

```
python main.py --gpu 0 --model arcfactoredmodel --config ./config/hyperparams_1.ini --name trial1 --actiontype evaluate
```

- The following files will be generated:
    - ./results/arcfactoredmodel.hyperparams_1.trial1.evaluation.arcs
    - ./results/arcfactoredmodel.hyperparams_1.trial1.evaluation.json

