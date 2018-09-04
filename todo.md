# model:

* two vocabularies, and the one for decoder is small and is the head of that of the encoder which is much bigger. The smaller vocabulary is the same as a part of the bigger one.

## the encoder

* first layer is a bilstm
* add a gate/filter layer for each state
* concatenate all matrices belongs to the same clause

* the second block contains several layers of cnn and a gate layer atop of the cnn stack

* the third block is for the sentence abstraction

## decoder
* separate the copy and attention mechanism
* the attention mechanism doesn't connect the word state
* the copy mechanism doesn't connect the top block
* noise should be injected to the labels 

# data processing
## preprocessing

  * replace unprintable characters with space
  ```vim
  :UnicodeToggle
  :%s/[^[:print:]]/ /g
  :UnicodeToggle
  ```
  * period connecting two words should be separated
      \.[A-Z]
  * replace all unicode escape string in the for \u[a-f0-9]{4} as UNCD_[a-f0-9]{4}, and make it back for apostrophes
  ```vim
  :%s/\v\\u([a-f0-9]{4})/ UNCD_\1 /g
  ```
  * the *pos* tag for each word, make the tag using unidecode to transliterate all non-ascii to ascii(decoded_unicode), then march the tags to the corresponding items; and get the *tf-idf*, *NER* for each word
    use unidecode transfer UNCD_[a-f0-9]{4} to UNCD_[a-f0-9]{4}
    original file: ascii string
    middle file: unicode string which is tokenized and contains all information: tag, tf_idf, pos, clause indecies and sentence indecies

  * the tf-idf score is obtained after stemming, and stop words are given lowest scores
  * do NP chunking and segment a sentence into noun phrases, obtain the phrase index
      the part index is the serial number of which clause the word belonging to 
  * in the same vein, get the sentence index for each clause, and the paragraph index for each sentence
  * substitue all links with a mark
  * uppercase and lower case issue: only NNP and NNPS don't need to be lowered
  * TODO: see if someother infor can be used
  * TODO: manually delete samples with too much non-english characters
  * TODO: consider if the stem of a world should be included in the embedding, or if a vocabulary of it is needed
    lexeme can reduce the dimension of the vector space representation, so it is necessary
    https://arxiv.org/pdf/1209.3126v1.pdf
  * [x] embedding: lemmatization + a_small_num * []. don't relate the lemmatization to embedding
  * use google embedding

### unicode problem
  * abandon those lines of which the abstract contain non-English, because the non-english characters in validation set are only chinese puncs and some chars of non sense. 
  * since the abstracts are written by human then there should be no unicode value in the the targets then all puncs, both in the contents and in the abstracts should be decoded into ascii puncs
  * all non ascii characters should be replaced by UNK except puncs. and continuous unks should be combined as one, and if a whole sentence contains only unks ignore it
  * \ua is a special tag not a unicode mark
  * TODO: to test if puncs can be removed, to see if it can affect the score
  * TODO: the generated first character doesn't need to be upcased, upcase it afterwards
  * remove \uac mess by hand

## steps
  * divide sentences
      \.[A-Z]|\\n|
