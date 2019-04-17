# ReCode
## Implementation of [Retrieval-Based Neural Code Generation](http://aclweb.org/anthology/D18-1111)

## BibTex
```
# coding=utf-8

@InProceedings{D18-1111,
  author = 	"Hayati, Shirley Anugrah
		and Olivier, Raphael
		and Avvaru, Pravalika
		and Yin, Pengcheng
		and Tomasic, Anthony
		and Neubig, Graham",
  title = 	"Retrieval-Based Neural Code Generation",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"925--930",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1111"
}
```

## Usage

Train a model from the beginning

```
./train.sh  <hs|django>
```

Use pre-trained model

```
./run_trained_model.sh <hs|django>
```

## File List

- train.sh                            # script for pre-training AST generative model and evaluation
- run_trained_model.sh     # script for running retrieval model based on pre-trained AST model
- run_interactive.sh           # run interactive_mode.py
- alignments.py                 # script for aligning input and retrieved descriptions
- astnode.py                      # contains  a node class for an AST
- code_gen.py                   # main code for training, evaluating, generating alignments
- components.py               # contains code for conditional LSTM and pointer net for copying
- config.py                         # configurations
- dataset.py                       # preprocessing code for dataset, such as setting vocabularies
- evaluation.py                  # evaluate retrieval model and save the output
- interactive_mode.py       # allows users to enter any query and outputs corresponding code
- learner.py                       # main code for managing training process
- main.py                          # contains utilities for parsing input
- parser.py                        # for testing parser
- parser_hiro.py                # print AST into readable form
- postprocess.py               # process outputs into a better form
- README                        # this README
- retrieval.py                      # helper function to get ngram sentences  
- retrievalmodel.py            # pytorch code for retrieval model ReCode
- simi.py                            # helper function to calculate sentence similarities
- nn/*      # pytorch code for vanilla LSTM/BiLSTM encoder and decoder with attention
- seq2seq/*                       # pytorch code for vanilla seq2seq model: decoder and encoder 
- lang/*                              #preprocessing code for parsing an AST tree 

