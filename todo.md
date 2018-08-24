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
  * the pos tag for each word, make the tag using unidecode to transliterate all non-asscii to asscii(decoded_unicode), then march the tags to the corresponding items; and get the tf-idf, NER for each word
    use unidecode transfer UNCD_[a-f0-9]{4} to UNCD_[a-f0-9]{4}_uniCharacter_decodedUnicode_DCNU
    original file: asscii string
    middle file: unicode string which is tokenized and contains all information: tag, tf_idf, pos, clause indecies and sentence indecies

  * get the clause index for each word using the unicode character
      the clause index is the serial number of which clause the word belonging to 
  * in the same vein, get the sentence index for each clause, and the paragraph index for each sentence

  * substitue all links with a mark

  * uppercase and lower case issue

## processing
  * divide sentences
      \.[A-Z]|\\n|
    
