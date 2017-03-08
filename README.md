
# data_science

> seeing is believing. A witty saying proves nothing.

> "When solving a problem of interest, do not solve a more general problem as an intermediate step." (Vladimir Vapnik)

Events: I will put word cloud for that.

EMNLP 2017: http://noisy-text.github.io/2017/

## NLPStan reading

- http://nlp.stanford.edu/read/
- NLP dataset: https://github.com/niderhoff/nlp-datasets

## LXMLS16:

- http://lxmls.it.pt/2016/Deep-Neural-Networks-Are-Our-Friends.pdf
- http://lxmls.it.pt/2016/lxmls-dl2.pdf

## VietAI

- Quoc Le (Google Brain): http://cs.stanford.edu/~quocle/
- Thang Luong (Google Brain): http://t.co/3zNHouUn
- Dustin (Columbia) http://dustintran.com/
- Thien (NYU) http://www.cs.nyu.edu/~thien/
- Hieu Pham (CMU) https://www.quora.com/profile/Hieu-Pham-20
- Ken Tran (Microsofts) http://www.kentran.net/
- Laurent Dinh (MILA):https://laurent-dinh.github.io/about/
- Luong Hoang, Harvard: https://github.com/lhoang29/recurrent-entity-networks

## My SOTA

- My ATIS: sequence tagging, nb of params: 324335, bi-LSTM
- Quore question duplicate detection: Accuracy 85% on Wang's test

```
 - best F1 score: 94.92/94.64
 - train scores: 97.5446666667/96.17
 - val scores: 93.664/92.94
```

## Yandex

- https://github.com/ddtm/dl-course
- https://github.com/vkantor/MIPT_Data_Mining_In_Action_2016/tree/master/trends
- https://github.com/yandexdataschool/Practical_RL
- https://github.com/yandexdataschool/HSE_deeplearning

## ICLR 2017 Review

- if you wanna turn LSTM, it's worth to read (from Socher): https://arxiv.org/pdf/1611.05104v2.pdf


## LearningNewThingIn2017

- Torch/Lua (Facebook/HarvardNLP): http://nlp.seas.harvard.edu/code/, http://cs287.fas.harvard.edu/
- TF/Python (Google/Stanford): https://github.com/BinRoot/TensorFlow-Book
- cs287: https://github.com/CS287/Lectures

## Conf events
- Coling 2016, Osaka Japan: http://coling2016.anlp.jp/
- ICLR 2017, Apr in France: http://www.iclr.cc/doku.php?id=ICLR2017:main&redirect=1
- open review: http://openreview.net/group?id=ICLR.cc/2017/conference

## NIPs 2016 slides

- https://github.com/hindupuravinash/nips2016
- Ian GAN tut: http://www.iangoodfellow.com/slides/2016-12-9-gans.pdf
- Ng nuts and bolts: https://www.dropbox.com/s/dyjdq1prjbs8pmc/NIPS2016%20-%20Pages%202-6%20(1).pdf
- variational inference: http://www.cs.columbia.edu/~blei/talks/2016_NIPS_VI_tutorial.pdf


## Theano based DL applications

- https://news.ycombinator.com/item?id=9283105

## learn to learn: algos optimization

- sgd and friends: http://cs231n.github.io/neural-networks-3/#update
- overview of gd: http://sebastianruder.com/optimizing-gradient-descent/
- https://github.com/fchollet/keras/issues/898
- I used to choose adam and rmsprop with tuning lr and batch size.


Pin: 

- semantic scholar: https://www.semanticscholar.org/
- grow a mind: http://web.mit.edu/cocosci/Papers/tkgg-science11-reprint.pdf
- trendingarxiv: http://trendingarxiv.smerity.com/
- https://github.com/andrewt3000/DL4NLP
- Natural languague inference NLI: https://github.com/Smerity/keras_snli
- ACL: http://www.aclweb.org/anthology/P/P16/

Data type: NOQ

- Nominal (N):cat, dog --> x,o | vis: shape, color
- Ordinal (O): Jan - Feb - Mar - Apr | vis: area, density
- Quantitative (Q): numerical 0.42, 0.58 | vis: length, position

People:

- Graham CMU: http://www.phontron.com/teaching.php, https://github.com/neubig/

Fin data:

- Reuters 8M (2007-2016): https://github.com/philipperemy/Reuters-full-data-set.git
- Bloomberg https://github.com/philipperemy/financial-news-dataset
- stocktwits: https://github.com/goodwillyoga/E107project/tree/master/pooja/data

Projects:

- https://github.com/THEdavehogue/glassdoor-analysis

Wikidata:

- https://github.com/VladimirAlexiev/VladimirAlexiev.github.io/blob/master/CH-names/README.org
- https://github.com/VladimirAlexiev/VladimirAlexiev.github.io/tree/master/CH-names

Cartoons & Quotes:

- "cause you know sometimes words have two meanings" led zeppelin
- http://stats.stackexchange.com/questions/423/what-is-your-favorite-data-analysis-cartoon?newsletter=1&nlcode=231076%7C1179

Books:

- http://neuralnetworksanddeeplearning.com/index.html
- u.cs.biu.ac.il/~yogo/nnlp.pdf

Done:

1. EMNLP 2016, Austin, 2-4 Nov: http://www.emnlp2016.net/tutorials.html#practical

- Dynet (CMU: https://t.co/nSCkBt0i0F
- lifelong ML (Google): http://www.emnlp2016.net/tutorials/chen-liu-t3.pdf
- Markov logic for scalable joint inference: http://www.emnlp2016.net/tutorials/venugopal-gogate-ng-t2.pdf
- good summary of sentiment analysis with NN (Singapore): http://www.emnlp2016.net/tutorials/zhang-vo-t4.pdf
- structure prediction (POS, NER)(Singapore): http://www.emnlp2016.net/tutorials/sun-feng-t6.pdf

- BADLS: 2 day conference at Stanford university

day 1: 

- Hugo(Twitter): Feed forward NN
- Kartpathy(OpenAI): Convnet
- Socher(MetaMind): NLP = word2vec/glove + GRU + MemNet
- Tensorflow tut: from 5:55:49
- Ruslan: Deep Unsup Learning: from 7:10:39
- Andrew Ng: Nuts and bolts in applied DL from 9:09:46

day 2: 

- Schulman: RL from 06:40
- Pascal(MILA): theano, from 1:52:03
- ASR from 4:01:11
- NN with Torch from 5:49:32, https://github.com/alexbw/bayarea-dl-summerschool
- seq2seq learning, Quoc Le: from 7:03:44
- Bengio: Foundations and challenges in DL, from 9:01:14

- data fest: https://alexanderdyakonov.wordpress.com/
- 8,9,12,13 Sept: data science week: http://dsw2016.datascienceweek.com/
- KDD 2016: http://www.kdd.org/kdd2016/
- ACL 2016, Berlin, 7-12 Aug: http://acl2016.org/index.php?article_id=60

AI mistakes:

- napalm girl: https://techcrunch.com/2016/09/12/facebook-employees-say-deleting-napalm-girl-photo-was-a-mistake/
- fine for his car shadow: http://www.independent.co.uk/news/world/europe/russian-driver-fined-car-shadow-moscow-a7225146.html
- human on motorcycle: http://cs.stanford.edu/people/karpathy/deepimagesent/generationdemo/ 

Keras:

- image classification with vgg16: http://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/
- hualos, keras viz: https://github.com/fchollet/hualos
- https://github.com/dylandrover/keras_tutorial/blob/master/keras_tutorial/keras_deck.pdf
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py
- model zoo:https://github.com/tensorflow/models
- music auto tag: https://github.com/keunwoochoi/music-auto_tagging-keras
- expose API: https://github.com/samjabrahams/inception-resnet-flask-demo

NLP:

- https://github.com/attardi/deepnl
- https://github.com/biplab-iitb/practNLPTools
- http://ml.nec-labs.com/senna/
- LSTM + CNN char on NER: https://transacl.org/ojs/index.php/tacl/article/viewFile/792/202

Apps:

- https://github.com/fginter/w2v_demo
- http://bionlp-www.utu.fi/wv_demo/
- 3top: https://github.com/3Top/word2vec-api
- next wave of nn: http://www.nextplatform.com/2016/09/14/next-wave-deep-learning-applications/
- labeling tools: http://cs.stanford.edu/people/karpathy/ilsvrc/
- deep art: https://deepart.io/hire/kzXhuUPf/
- text sum: http://esapi.intellexer.com/Summarizer
- http://www.deeplearningpatterns.com/doku.php/applications
- mt: http://104.131.78.120/
- rnn: http://www.cs.toronto.edu/~ilya/fourth.cgi?prefix=I+have+a+dream.+&numChars=150
- chatbot: http://sumve.com/firesidechat/
- text vis: http://slanglab.cs.umass.edu/topic-animator/
- music auto tag: https://github.com/keunwoochoi/music-auto_tagging-keras
- deep image sent: http://cs.stanford.edu/people/karpathy/deepimagesent/rankingdemo/

German word embedding:

- pretrained: http://devmount.github.io/GermanWordEmbeddings/
- vis: pca, tsne: https://github.com/devmount/GermanWordEmbeddings/blob/master/code/pca.ipynb

PyGotham:

- textacy: http://michelleful.github.io/code-blog/2016/07/23/nlp-at-pygotham-2016/
- nlp with keras, rnn, cnn
- https://github.com/drincruz/PyGotham-2016
- skipthought: https://libraries.io/github/LeavesBreathe/Sequence-To-Sequence-Generation-Skip-Thoughts-
- https://github.com/ryankiros/skip-thoughts
- doc sum: http://mike.place/talks/pygotham/#p1

Journalist LDA and ML:

- http://knightlab.northwestern.edu/2015/03/10/nicar-2015-machine-learning-lessons-for-journalists/
- summary on hanna wallach https://docs.google.com/document/d/1kIIzBAF9T9Zu99i0DU9akIajvYZ-CfHeBFVBhIJyEY8/edit?pref=2&pli=1
- http://www.cs.ubc.ca/~murphyk/MLbook/pml-toc-22may12.pdf
- http://slides.com/stevenrich/machine-learning#/18
- https://github.com/cjdd3b/nicar2015/tree/master/machine-learning
- https://github.com/cjdd3b/fec-standardizer

Europython:

- http://kjamistan.com/i-hate-you-nlp/
- https://github.com/adewes/machine-learning-chinese
- https://github.com/GaelVaroquaux/my_topics
- https://github.com/arnicas/nlp_elasticsearch_reviews

Scipy 2016:

- http://scipy2016.scipy.org/ehome/146062/332963/

Performance Evaluation(PE):

- book ELA: http://www.cambridge.org/us/academic/subjects/computer-science/pattern-recognition-and-machine-learning/evaluating-learning-algorithms-classification-perspective
- slides: http://www.icmla-conference.org/icmla11/PE_Tutorial.pdf
- bayesian hypothesis testing: http://ipg.idsia.ch/preprints/corani2015c.pdf 

Hypothesis testing

- http://bebi103.caltech.edu/2015/tutorials/t6b_frequentist_hypothesis_testing.html
- central limit theorem: http://nbviewer.jupyter.org/github/mbakker7/exploratory_computing_with_python/blob/master/notebook_s3/py_exp_comp_s3_sol.ipynb
- hypothesis testing and p value: http://vietsciences.free.fr/khaocuu/nguyenvantuan/bieudoR/ch7-kiemdinhgiathiet.htm

Metrics:

- http://users.dsic.upv.es/~dpinto/duc/RougeLin.pdf

Rock, Metal and NLP:

- http://www.deepmetal.io/
- https://github.com/ijmbarr/metal_models
- http://www.degeneratestate.org/posts/2016/Sep/12/heavy-metal-and-natural-language-processing-part-2/
- http://www.degeneratestate.org/posts/2016/Apr/20/heavy-metal-and-natural-language-processing-part-1/

Financial:

- https://github.com/johnymontana/NewzTrader_AI_project

Twitter:

- http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

Deep Learning Frameworks/Toolkits:

- Tensorflow
- Torch
- Theano
- Keras
- Dynet
- CNTK

ElasticSearch + Kibana:

- install ES 2.4 + Kibana: default sense in console 5601
- http://ghostweather.slides.com/lynncherny/deck


Attention based:

- decomposable attention: https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment
- customized lstm with attention: http://benjaminbolte.com/blog/2016/keras-language-modeling.html
- vis + cnn + lstm: https://blog.heuritech.com/2016/01/20/attention-mechanism/


ResNet: Residual Networks

- http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html
- how deep Vgg 16,19 vs 152 200 layers: https://www.reddit.com/r/MachineLearning/comments/4cmcfs/how_can_resnet_cnn_go_deep_to_152_layers_and_200/
- http://www.slideshare.net/Textkernel/practical-deep-learning-for-nlp


Sentiment

- dataset: 1.6M: https://docs.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download
- quandl: https://github.com/kszela24/options-daily
- stocktwit: http://stocktwits.com/symbol/FINL
- https://github.com/jssandh2/Stock_Search_Engine
- https://www.quantopian.com/posts/crowd-sourced-stock-sentiment-using-stocktwits

----

# Timeline

08.03

- CMU RF and control course: https://katefvision.github.io/
- https://www.slideshare.net/JasonKessler/turning-unstructured-content-into-kernels-of-ideas/52
- norvig ngram: http://norvig.com/ngrams/


07.03

- https://www.slideshare.net/JasonKessler/turning-unstructured-content-into-kernels-of-ideas/52
- https://arxiv.org/pdf/1703.00565.pdf
- https://jasonkessler.github.io/st-sim.html
- Dr Bao H.T JAIST: http://www.jaist.ac.jp/~bao/VIASM-SML/Lecture/L1-ML%20overview.pdf
- Khanh UMD: https://github.com/khanhptnk?tab=repositories

06.03

- http://campuspress.yale.edu/yw355/deep_learning/
- https://github.com/georgeiswang/Keras_Example
- https://github.com/thomasj02/DeepLearningProjectWorkflow
- https://tensorflow.github.io/serving/docker.html
- Deep learning in NLP: http://campuspress.yale.edu/yw355/deep_learning/

05.03

- fcholet: xception https://arxiv.org/pdf/1610.02357.pdf

04.03

- https://github.com/jfsantos/TensorFlow-Book
- https://github.com/jfsantos/keras-tutorial/blob/master/notebooks/5%20-%20Improving%20generalization%20with%20regularizers%20and%20constraints.ipynb


02.03

- https://explosion.ai/blog/supervised-similarity-siamese-cnn
- https://github.com/TeamHG-Memex/eli5/blob/master/README.rst
- https://github.com/cemoody/topicsne?files=1
- http://smerity.com/articles/2017/deepcoder_and_ai_hype.html

01.03

- http://smerity.com/articles/2017/deepcoder_and_ai_hype.html
- Twitter NER annotation: https://docs.google.com/document/d/12hI-2A3vATMWRdsKkzDPHu5oT74_tG0-PPQ7VN0IRaw/edit
- WNUT 19, Japan, result: https://noisy-text.github.io/2016/pdf/WNUT19.pdf
- pytorch vs keras/tf: https://www.reddit.com/r/MachineLearning/comments/5w3q74/d_so_pytorch_vs_tensorflow_whats_the_verdict_on/
- quora duplicate question detection: accuracy 1%(84.8) higher but 100x params than my model: https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/deepnet.py
- https://github.com/chiphuyen/tf-stanford-tutorials?files=1
- pretrained fasttext on wikipedia: https://github.com/facebookresearch/fastText

28.02

- https://github.com/uclmr/emoji2vec/blob/master/TwitterClassification.ipynb
- http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
- https://blog.mariusschulz.com/2014/06/03/why-using-in-regular-expressions-is-almost-never-what-you-actually-want

27.02

- random walk -> graph -> node2vec: http://www.kdd.org/kdd2016/subtopic/view/node2vec-scalable-feature-learning-for-networks
- URL2VEC: http://www.newfoundland.nl/wp/?p=112
- 5 diseases of doing science: http://www.sciencedirect.com/science/article/pii/S104898431730070X
- recommended book: https://www.amazon.com/Language-Processing-Perl-Prolog-Implementation/
- Martin DL without PHD: https://github.com/martin-gorner/tensorflow-mnist-tutorial
- https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0
- https://docs.google.com/presentation/d/18MiZndRCOxB7g-TcCl2EZOElS5udVaCuxnGznLnmOlE/pub?slide=id.p
- https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.p

26.02

- https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b#.diz6kjaus
- https://github.com/gkamradt/Lessons-Learned-Data-Science-Interviews/blob/master/Lessons%20Learned%20-%20Data%20Science%20Interviews.pdf

25.02

- gensim 1.0: https://rare-technologies.com/gensim-switches-to-semantic-versioning/
- https://www.slideshare.net/AhmadQamar3/using-deep-neural-networks-for-fashion-applications

24.02

- how to init uniform (-b,b), summerschool of marek http://www.marekrei.com/blog/26-things-i-learned-in-the-deep-learning-summer-school/
- Beam preprocessing: https://research.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html
- https://github.com/offbit/char-models/blob/master/doc-rnn2.py

23.02

- http://affinelayer.com/pixsrv/
- https://github.com/affinelayer/pix2pix-tensorflow#datasets-and-trained-models


22.02

- https://github.com/offbit/char-models
- https://offbit.github.io/how-to-read/
- https://hackernoon.com/learning-ai-if-you-suck-at-math-p4-tensors-illustrated-with-cats-27f0002c9b32#.xqpspe69f
- Beam search, NN tut from Quoc Le: https://cs.stanford.edu/~quocle/tutorial2.pdf
- marek sequence tagger: https://github.com/marekrei/sequence-labeler

21.02

- https://github.com/marekrei/sequence-labeler
- markrei word + char attention: http://www.marekrei.com/blog/
- datalab: https://github.com/googledatalab/
- https://tw.pycon.org/2017/en-us/speaking/cfp/

20.02

- https://github.com/ZhitingHu/logicnn
- http://www.cs.cmu.edu/~zhitingh/data/acl16harnessing_slides.pdf
- Lample: https://arxiv.org/pdf/1603.01360.pdf, https://github.com/glample/tagger
- stacked NN LSTM: https://github.com/clab/stack-lstm-ner
- https://github.com/napsternxg/DeepSequenceClassification/blob/master/model.py
- chatbot: https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm
- keras crf https://github.com/pressrelations/keras/blob/98b2bb152b8d472150a3fc4f91396ce7f767bed9/examples/conll2000_bi_lstm_crf.py
- Ma Xue, CMU: best paper in ACL 2016, Germany https://github.com/XuezheMax/LasagneNLP
- rnn+cnn+crf: https://arxiv.org/pdf/1603.01354.pdf
- https://github.com/napsternxg/DeepSequenceClassification/blob/master/model.py
- https://github.com/pth1993/vn_spam_sms_filtering/blob/master/src/sms_filtering.py
- https://data36.com/wp-content/uploads/2016/08/practical_data_dictionary_final_data36_tomimester_published.pdf

19.02

- scikit plot: https://github.com/reiinakano/scikit-plot


18.02

- really cool Francis: https://github.com/frnsys/
- ai notes: http://frnsys.com/ai_notes/ai_notes.pdf
- brilliant wrong, ROC explanation: http://arogozhnikov.github.io/2015/10/05/roc-curve.html
- yandex MLSchool in Londo: https://github.com/yandexdataschool/MLatImperial2017/

17.02

- RNNs bag of applications: http://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/RNN.pdf
- BiMPM https://arxiv.org/pdf/1702.03814.pdf
- TextSum step by step: http://www.fastforwardlabs.com/luhn/
- https://keon.io/rl/deep-q-learning-with-keras-and-gym/
- https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2#.ny8j80fl3
- big 5 for DS: https://www.quora.com/How-do-you-judge-a-good-Data-scientist-with-just-5-questions
- keon: https://github.com/keon/awesome-nlp
- quid: word2vec + wikipedia: https://quid.com/feed/how-quid-improved-its-search-with-word2vec-and-wikipedia?utm_content=42445351&utm_medium=social&utm_source=twitter
- https://gist.github.com/asmeurer/5843625

16.02

- market2vec: https://github.com/talolard/MarketVectors/blob/master/preparedata.ipynb
- anything2vec: https://gist.github.com/nzw0301/333afc00bd508501268fa7bf40cafe4e
- https://github.com/bradleypallen/keras-movielens-cf
- https://www.slideshare.net/t_koshikawa?utm_campaign=profiletracking&utm_medium=sssite&utm_source=ssslideview
- https://github.com/lipiji/App-DL
- http://www.slideshare.net/LimZhiYuanZane/deep-learning-for-stock-prediction
- https://github.com/kh-kim/stock_market_reinforcement_learning
- stock2vec: https://github.com/kh-kim/stock2vec
- deepwalk and word2vec: http://nadbordrozd.github.io/blog/2016/06/13/deepwalking-with-companies/
- http://m-mitchell.com/NAACL-2016/SemEval/SemEval-2016.pdf
- gandl: https://github.com/codekansas/gandlf
- predictive on stock trading with sentiment: http://www.kdnuggets.com/2016/01/sentiment-analysis-predictive-analytics-trading-mistake.html
- https://github.com/bradleypallen/keras-emoji-embeddings
- https://github.com/bradleypallen/keras-quora-question-pairs/blob/master/README.md
- DESM: https://www.microsoft.com/en-us/research/project/dual-embedding-space-model-desm/

15.02

- sentiment analysis on Super Bowl: http://blog.aylien.com/sentiment-analysis-of-2-2-million-tweets-from-super-bowl-51/
- spacy advanced text analysis: https://github.com/JonathanReeve/advanced-text-analysis-workshop-2017/blob/master/advanced-text-analysis.ipynb 
- pytorch: https://github.com/vinhkhuc/PyTorch-Mini-Tutorials
- Quora engineering: https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Space bag of nns: https://explosion.ai/blog/quora-deep-text-pair-classification
- AUC 0.875 http://analyzecore.com/2017/02/08/twitter-sentiment-analysis-doc2vec/

14.02

- event detection, extraction, triggering, mention: https://github.com/anoperson/jointEE-NN
- batch renorm, due to sensitivity of batch size, initiation: https://arxiv.org/pdf/1702.03275.pdf
- https://github.com/bmitra-msft/Demos/blob/master/notebooks/DESM.ipynb
- nn for document ranking, mistra, ms cntk: https://github.com/bmitra-msft/NDRM
- TFDevSummit: https://events.withgoogle.com/tensorflow-dev-summit/watch-the-videos/#content

13.02

- Quora siamese: https://github.com/erogol/QuoraDQBaseline

12.02

- http://www.slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017

10.02

- kerlym: https://github.com/osh/kerlym
- ICLR 17: https://amundtveit.com/2016/11/12/deep-learning-for-natural-language-processing-iclr-2017-discoveries/
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
- all but of the top, pca on word2vec: https://arxiv.org/pdf/1702.01417.pdf
- https://github.com/peter3125/sentence2vec

08.02

- polarised term for document anonymisation: https://ddu1.github.io/Anonymization/
- oxford course: https://github.com/oxford-cs-deepnlp-2017/lectures
- tf fold: dynamic batching: https://research.googleblog.com/2017/02/announcing-tensorflow-fold-deep.html
- https://www.insight-centre.org/sites/default/files/publications/newhorizons_online.pdf
- https://github.com/chsasank/Traffic-Sign-Classification.keras/blob/master/Traffic%20Sign%20Classification.ipynb

07.02

- openrefine: http://alexpetralia.com/posts/2015/12/14/the-problem-with-openrefine-clean-vs-messy-data
- https://www.linkedin.com/pulse/keras-neural-networks-win-nvidia-titan-x-abhishek-thakur
- deep q learning with keras and gym: https://keon.io/rl/deep-q-learning-with-keras-and-gym/
- structured attention, Yoon Kim and Hoang Luong: https://github.com/harvardnlp/struct-attn
- understanding DL requires rethinking generalisation: https://openreview.net/pdf?id=Sy8gdB9xx
- GAN: https://github.com/osh/KerasGAN

06.02

- http://lxmls.it.pt/2016/LxMLS2016.pdf
- http://www.cs.umb.edu/~twang/file/tricks_from_dl.pdf
- https://svn.spraakdata.gu.se/repos/richard/pub/ml2016_web/LT2306_2016_example_solution.pdf
- https://svn.spraakdata.gu.se/repos/richard/pub/ml2015_web/l7.pdf
- https://chsasank.github.io/spoken-language-understanding.html
- ML4NLP: http://stp.lingfil.uu.se/~shaooyan/ml/nn.part2.pdf
- Topic Modeling for extracting key words: http://bugra.github.io/work/notes/2017-02-05/topic-modeling-for-keyword-extraction/
- Google Scraper: https://github.com/NikolaiT/GoogleScraper
- Richard Johanson: https://svn.spraakdata.gu.se/repos/richard/pub/ml2015_web/l7.pdf
- https://code.facebook.com/posts/457605107772545/under-the-hood-building-accessibility-tools-for-the-visually-impaired-on-facebook/
- l2svm outperforms softmax: https://arxiv.org/pdf/1306.0239v4.pdf
- xent vs hinge loss: http://cs231n.github.io/linear-classify/
- https://github.com/nzw0301/keras-examples/blob/master/Skip-gram-with-NS.ipynb
- model zoo pytorch: https://github.com/Cadene/tensorflow-model-zoo.torch
- quora question pair: http://www.forbes.com/sites/quora/2017/01/30/data-at-quora-first-quora-dataset-release-question-pairs/#3d052ef475cb
- Psychometric, CA and Trump: https://motherboard.vice.com/en_us/article/how-our-likes-helped-trump-win


27.1

- https://github.com/bbelderbos/Codesnippets/tree/master/python
- https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.htm

26.1

- https://jaan.io/food2vec-augmented-cooking-machine-intelligence/
- http://multithreaded.stitchfix.com/blog/2017/01/23/scaling-ds-at-sf-slides-from-ddtexas/
- https://docs.docker.com/docker-for-mac/
- https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#1
- https://petewarden.com/

25.1

- question duplication of Quora: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
- stats for hackers code: https://github.com/croach/blog/tree/master/content
- http://multithreaded.stitchfix.com/blog/2017/01/23/scaling-ds-at-sf-slides-from-ddtexas/

24.1

- wordrank: http://deliprao.com/archives/124
- code: https://bitbucket.org/shihaoji/wordrank
- https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/WordRank_wrapper_quickstart.ipynb
- https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/WordRank_wrapper_quickstart.ipynb
- https://github.com/parulsethi/gensim/blob/wordrank_wrapper/docs/notebooks/Wordrank_comparisons.ipynb
- https://rare-technologies.com/wordrank-embedding-crowned-is-most-similar-to-king-not-word2vecs-canute/

23.1

- nlp terms for novice: http://www.datasciencecentral.com/profiles/blogs/10-common-nlp-terms-explained-for-the-text-analysis-novice?utm_content=buffer172af&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer
- blockchain: https://opendatascience.com/blog/what-is-the-blockchain-and-why-is-it-so-important/
- nbgrader: https://github.com/jupyter/nbgrader
- Adversarial ML: https://mascherari.press/introduction-to-adversarial-machine-learning/
- 4 questions for G. Hinton: https://gigaom.com/2017/01/16/four-questions-for-geoff-hinton/
- Debug in TF: https://wookayin.github.io/TensorflowKR-2016-talk-debugging/#1

20.1

- demysify DS: https://docs.google.com/presentation/d/1N3KhPA--cQNjF9mD4Z4IzjKKFdwq1Ff6wQ6NN102uIk/edit#slide=id.g1be386a8a6_0_21
- ML on mobile: http://alexsosn.github.io/ml/2015/11/05/iOS-ML.html
- https://www.bignerdranch.com/blog/use-tensorflow-and-bnns-to-add-machine-learning-to-your-mac-or-ios-app/
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples
- https://github.com/dennybritz/sentiment-analysis


19.1

- Facebook again, pytorch: http://pytorch.org/
- https://rare-technologies.com/new-gensim-feature-author-topic-modeling-lda-with-metadata/
- pointer network: https://github.com/devsisters/pointer-network-tensorflow

18.1

- http://blog.dennybritz.com/2017/01/17/engineering-is-the-bottleneck-in-deep-learning-research/
- ml for practitioner: http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf
- write dl/nn from scratch: https://github.com/dmlc/minpy

17.1

- improve headlines with salient words and seo score: http://www-personal.umich.edu/~tdszyman/misc/nlpmj16.pdf
- text summarisation: http://www-personal.umich.edu/~tdszyman/misc/summarization15.pdf
- word embedding over time: http://www-personal.umich.edu/~tdszyman/misc/InsightSIGNLP16.pdf
- victor DS politech in France: https://github.com/Vict0rSch/data_science_polytechnique
- Thien NYU: http://www.cs.nyu.edu/~thien/
- tonymooori: https://github.com/TonyMooori/studying
- learning theory: https://web.stanford.edu/class/cs229t/notes.pdf
- time series predictions: http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/

16.1

- Edward Dustin Tran in TF already, so cool: https://arxiv.org/pdf/1701.03757v1.pdf
- keras in tensorflow now on. @fchollet informed on Twitter.
- squeezednet = tiny alexnet (5MB) https://github.com/rcmalli/keras-squeezenet
- won $5k: https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc#.9yb31nsm4
- https://github.com/karoldvl/paper-2015-esc-convnet/blob/master/Code/Results.ipynb

15.1

- deep spell code: https://github.com/MajorTal/DeepSpell
- draw svg in jupyter: https://github.com/uclmr/egal
- sound classification with cnn: https://github.com/karoldvl/paper-2015-esc-convnet

14.1

- https://medium.com/@majortal/deep-spelling-9ffef96a24f6
- line bot + rnn + tf, vanhuyz: https://github.com/vanhuyz/line-sticker-bot
- https://github.com/Vict0rSch/deep_learning/tree/master/keras
- https://github.com/openai/pixel-cnn
- AWS Lambda: http://blog.matthewdfuller.com/p/aws-lambda-pricing-calculator.html
- deep text corrector: http://atpaino.com/2017/01/03/deep-text-correcter.html
- https://github.com/dhwajraj/deep-text-classifier-mtl

13.1

- convlstm: https://github.com/carlthome/tensorflow-convlstm-cell
- GAN and RNN: https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/
- generate sentences from continuous space: https://arxiv.org/pdf/1511.06349v2.pdf
- How to train your Gen. model: Sampling, likelihood or adversary


12.1

- https://www.raywenderlich.com/126063/react-native-tutorial
- ml practitioners: https://news.ycombinator.com/item?id=10954508
- spotify word2vec: https://douweosinga.com/projects/marconi?song1_id=45yEy5WJywhJ3sDI28ajTm&song2_id=
- https://github.com/DOsinga/marconi/blob/master/train_model.py
- True| Good | Kind | Useful | Relevant | Necessary https://www.quora.com/What-is-Triple-Filter-test-of-Socrates
- https://www.youtube.com/watch?v=ifYfJdo27_k
- student note: https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-3-Natural-Language-Processing

11.1

- ggplot2 in R: http://sharpsightlabs.com/blog/mapping-vc-investment/
- TF 1.0, mature. https://opendatascience.com/blog/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
- NN semantic encoder: https://github.com/pdasigi/neural-semantic-encoders/blob/master/nse.py
- DL in NN, overview: https://arxiv.org/pdf/1404.7828v4.pdf
- jurgen schmid: http://people.idsia.ch/~juergen/

10.1

- GDG NL: http://www.slideshare.net/RokeshJankie/introducing-tensorflow-the-game-changer-in-building-intelligent-applications
- https://github.com/ToferC/Twitter_graphing_python
- http://www.oujago.com/DL_more.html
- thiago DS at Yahoo: https://tgmstat.wordpress.com/
- deepstack playing poker: https://arxiv.org/pdf/1701.01724v1.pdf
- silly DL: https://news.ycombinator.com/item?id=13353941
- http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html
- AE for new molecule: http://www.impactjournals.com/oncotarget/index.php?journal=oncotarget&page=article&op=view&path[]=14073&pubmed-linkout=1

9.1

- xlingual embedding: https://levyomer.wordpress.com/2017/01/08/a-strong-baseline-for-learning-cross-lingual-word-embeddings-from-sentence-alignments/
- greg notebooks: https://github.com/gjreda/gregreda.com/tree/master/content/notebooks
- the periodic table of AI: http://ai.xprize.org/news/periodic-table-of-ai
- the same table of DL: http://www.deeplearningpatterns.com/doku.php/overview
- aylien text mining and analysis: Sebastien Ruder: https://arxiv.org/pdf/1609.02746v1.pdf
- DS as a freelancer from Greg Yhat: http://www.gregreda.com/2017/01/07/freelance-data-science-experience/

7.1

- how bayesian inference works: http://brohrer.github.io/how_bayesian_inference_works.html
- best vis projects in 2016: http://flowingdata.com/2016/12/29/best-data-visualization-projects-of-2016/
- https://flowingdata.com/2012/12/17/getting-started-with-charts-in-r/


5.1

- allenai biattflow: https://github.com/allenai/bi-att-flow
- fork guy: https://github.com/BinbinBian
- ICRL 17, DCNN: https://arxiv.org/pdf/1611.01604v2.pdf
- victor zhong: https://github.com/vzhong/posts-notebooks
- BN, if you wann gaussian, zero mean: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
- statsnlp https://github.com/uclmr/stat-nlp-book
- sota of qa: http://metamind.io/research/state-of-the-art-deep-learning-model-for-question-answering/

4.1

- dynet: CMU neural networks in C++: https://github.com/clab
- systran: https://arxiv.org/pdf/1610.05540v1.pdf
- punctuation normalisation: http://www.statmt.org/wmt11/normalize-punctuation.perl
- GAN in keras: https://github.com/osh/KerasGAN
- reinforcement learning in keras and gym: https://github.com/osh/kerlym
- ML 101 for DE: https://drive.google.com/drive/folders/0B3bb7xB2VOUBMW1LQjVYUlJNRFU

3.1

- variational for text processing: https://github.com/carpedm20/variational-text-tensorflow
- spotify CNN music classification: https://www.dropbox.com/s/22bqmco45179t7z/thesis-FINAL.pdf
- kaggle winning solution for whale detection: https://github.com/benanne
- https://github.com/zygmuntz?tab=repositories

2.1.17

- overfitting in life: http://tuanvannguyen.blogspot.com/2016/12/over-fitting-va-y-nghia-thuc-te-trong.html
- optimal stopping problem: https://plus.maths.org/content/solution-optimal-stopping-problem

31.12

- visualisation NLP: http://www.aclweb.org/anthology/N16-1082

30.12

- zero shot translation: https://techcrunch.com/2016/11/22/googles-ai-translation-tool-seems-to-have-invented-its-own-secret-internal-language/

29.12

- Music Tagging, CRNN https://arxiv.org/pdf/1609.04243v3.pdf
- Benmusic: http://www.bensound.com/
- event detection: http://anthology.aclweb.org/C/C14/C14-1134.pdf

28.12

- NIPs 2016, embedding projector: https://arxiv.org/pdf/1611.05469.pdf
- stats learning: https://web.stanford.edu/class/cs229t/notes.pdf
- http://www.normansoft.com/blog/index.html
- Tf projector is really cool: https://github.com/normanheckscher/mnist-tensorboard-embeddings/blob/master/mnist_t-sne.py
- Who to follow on Twitter in ML/DL: https://twitter.com/DL_ML_Loop/lists/deep-learning-loop/members
- How to learn? BPTT https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.sunmvqmsx

27.12

- deep learning with Torch: https://github.com/soumith/cvpr2015
- T7: https://github.com/soumith/cvpr2015/blob/master/cvpr-torch.pdf
- GPOD general purpose object detector: https://github.com/EvgenyNekrasov/gpod
- mckinseys: http://www.forbes.com/sites/louiscolumbus/2016/12/18/mckinseys-2016-analytics-study-defines-the-future-machine-learning
- gumbel add noise to sigmoid: https://github.com/yandexdataschool/gumbel_lstm
- fastai wordembedding: https://github.com/fastai/courses/blob/master/deeplearning1/nbs/wordvectors.ipynb

26.12

- spotify cnn: http://benanne.github.io/2014/08/05/spotify-cnns.html
- Gated RNN https://arxiv.org/pdf/1612.08083v1.pdf
- http://www.slideshare.net/SebastianRuder/nips-2016-highlights-sebastian-ruder
- monolingal dataset WMT 2014: http://www.statmt.org/wmt14/translation-task.html
- neural turing machine: https://github.com/shawntan/neural-turing-machines
- yandex ml school HSE: https://github.com/yandexdataschool/HSE_deeplearning


24.12

- Laurent Dinh: Density estimation https://docs.google.com/presentation/d/152NyIZYDRlYuml5DbBONchJYA7AAwlti5gTWW1eXlLM/
- Swiftkey, LM: https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/
- porting Theano to TF: https://medium.com/@sentimentron/faceoff-theano-vs-tensorflow-e25648c31800
- tractica: DL for retailer: https://www.tractica.com/automation-robotics/leveraging-deep-learning-to-improve-the-retail-experience/
- Effective Size: is Singaporean better in math than Vietnamese? if ES = 0.3, the overlap is near 90%, nothing to say in this Pisa's ranking.
- dracula: twitter POS utilised GATE: https://github.com/Sentimentron/Dracula/
- Business process with LSTM: https://arxiv.org/pdf/1612.02130v1.pdf


23.12

- https://bigdatauniversity.com/courses/deep-learning-tensorflow/

22.12

- https://quid.com/feed/how-quid-uses-deep-learning-with-small-data
- dl for coders: http://course.fast.ai/, notebooks here: https://github.com/fastai/courses
- encoder-decoder RNN: http://www.slideshare.net/ssuser77b8c6/reducing-the-dimensionality-of-data-with-neural-networks
- https://trello.com/b/rbpEfMld/data-science
- http://tuanvannguyen.blogspot.com/2016/12/yeu-to-nao-anh-huong-en-iem-pisa-2015.html

21.12

- https://github.com/napsternxg/TwitterNER
- news arxiv: https://news.google.com/newspapers?hl=en#F
- https://github.com/skillachie/binaryNLP
- https://github.com/skillachie/nlpArea51/blob/master/Financial_News_Text_Classification.ipynb
- http://www.kdnuggets.com/2016/12/machine-learning-artificial-intelligence-main-developments-2016-key-trends-2017.html

20.12

- http://opennmt.net
- neural relation extraction https://www.aclweb.org/anthology/P/P16/P16-1200.pdf
- claim classification: https://github.com/UKPLab/coling2016-claim-classification
- https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2016/2016_COLING_CG.pdf

19.12

- fasttext.zip https://arxiv.org/abs/1612.03651
- bi sequence classification: same SNLI, event detection: https://pdfs.semanticscholar.org/6f42/cb23262066b4034aba99bf674783ed6cac8b.pdf
- large scale contextual LSTM and NLP task: https://arxiv.org/pdf/1602.06291.pdf
- main advances in ML 2016, Xavier at Quora: https://www.quora.com/What-were-the-main-advances-in-machine-learning-artificial-intelligence-in-2016?


17.12

- https://github.com/jwkvam/bowtie

16.12

- tensorflow book with code: https://github.com/BinRoot/TensorFlow-Book
- trading with ML (Georgia university): https://www.udacity.com/course/machine-learning-for-trading--ud501

15.12

- deepbach: https://github.com/SonyCSL-Paris/DeepBach
- https://www.technologyreview.com/s/603137/deep-learning-machine-listens-to-bach-then-writes-its-own-music-in-the-same-style/
- http://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html?_r=0
- http://www.asimovinstitute.org/analyzing-deep-learning-tools-music/

14.12

- spacy vs nltk: https://gist.github.com/rschroll/61b20c41e984a963df2870cfc9e628ed
- psychometrics, precision marketing, privacy no longer: http://www.michalkosinski.com/
- 300+ ML projects from Stanford: http://cs229.stanford.edu/PosterSessionProgram.pdf
- NIPs 2016 codes: https://www.reddit.com/r/MachineLearning/comments/5hwqeb/project_all_code_implementations_for_nips_2016/
- Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences: https://github.com/dannyneil/public_plstm


13.12

- NIPs summary: http://beamandrew.github.io/deeplearning/2016/12/12/nips-2016.html
- how to choose batch size: https://github.com/karpathy/char-rnn, https://svail.github.io/rnn_perf/, http://axon.cs.byu.edu/papers/Wilson.nn03.batch.pdf
- https://github.com/lmthang/thesis

12.12

- Relation classification (RC) via data augmentation: https://arxiv.org/abs/1601.03651
- broader twitter NER: http://www.slideshare.net/leonderczynski/broad-twitter-corpus-a-diverse-named-entity-recognition-resource
- sequence classification such as NER, POS: https://github.com/napsternxg/DeepSequenceClassification
- arctic captions: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
- COLING 2016 from 13 to 16 Dec, Japan: https://github.com/napsternxg/TwitterNER, http://coling2016.anlp.jp/

11.12

- SRL and RC: https://github.com/jiangfeng1124/emnlp14-semi, http://ir.hit.edu.cn/~jguo/papers/coling2016-mtlsrc.pdf
- https://blog.insightdatascience.com/nips-2016-day-3-highlights-robots-that-know-cars-that-see-and-more-1ec958896791
- http://www.newsreader-project.eu/files/2012/12/NWR-D5-2-1.pdf
- http://nlesc.github.io/UncertaintyVisualization/
- http://ixa2.si.ehu.es/nrdemo/demo.php
- http://ir.hit.edu.cn/~jguo/papers/coling2016-mtlsrc.pdf

9.12

- if then learning: https://papers.nips.cc/paper/6284-latent-attention-for-if-then-program-synthesis.pdf
- reinforcement learning: https://github.com/DanielTakeshi
- NIPS 2016: https://github.com/mphuget/NIPS2016
- https://github.com/zelandiya/KiwiPyCon-NLP-tutorial
- http://www.wrangleconf.com/apac.html
- http://cs231n.github.io/aws-tutorial/
- clickbait F1 98, AUC 99, too good too be true: https://arxiv.org/pdf/1612.01340v1.pdf
- https://arxiv.org/abs/1606.04474
- https://github.com/deepmind/learning-to-learn

8.12

- hackermath: https://github.com/amitkaps/hackermath/blob/master/talk.pdf
- tensorboard: https://www.tensorflow.org/versions/master/how_tos/embedding_viz/index.html
- embedding projector: http://projector.tensorflow.org/ 
- dl4nlp at ukplab, Germany: https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar
- Filter bubble vs Info cascading, Eli Pariser: https://www.ted.com/talks/eli_pariser_beware_online_filter_bubbles

7.12

- tidy data in pandas: http://www.jeannicholashould.com/tidy-data-in-python.html
- graph db: https://blog.grakn.ai/adding-semantics-to-graph-databases-with-mindmapsdb-part-1-82022bbb3b1c
- https://github.com/mikonapoli
- reinforcement learninghttp, open ai://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf
- meal description and food tagging: https://pdfs.semanticscholar.org/5f55/c5535e80d3e5ed7f1f0b89531e32725faff5.pdf

6.12

- rationale cnn [keras] https://github.com/bwallace/rationale-CNN
- churn analysis, f1 75%, lr, svm hinge: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9849/9527
- thanapon noraset: https://northanapon.github.io/read/
- https://github.com/NorThanapon/adaptive_lm
- train general AI: https://openai.com/blog/universe/
- NIPS 2016 https://nips.cc/Conferences/2016/Schedule
- full ds notebook: https://github.com/donnemartin/data-science-ipython-notebooks
- Quoc Le, tut2: Autoencoder, CNN, RNN: http://ai.stanford.edu/~quocle/tutorial2.pdf
- Quoc Le, tut1: nonlinear classifier and backprop: http://ai.stanford.edu/~quocle/tutorial1.pdf
- Quoc Le, ex1: http://ai.stanford.edu/~quocle/exercise1.py
- https://alexanderdyakonov.wordpress.com/2016/12/04/сундуки-и-монеты/#more-4401

5.12

- semantic role labelings: https://blog.acolyer.org/2016/07/05/end-to-end-learning-of-semantic-role-labeling-using-recurrent-neural-networks/
- ml yearning: https://gallery.mailchimp.com/dc3a7ef4d750c0abfc19202a3/files/Machine_Learning_Yearning_V0.5_01.pdf
- stock embedding:https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02#.9q1d9hnai
- fast weights: https://github.com/ajarai

2.12

- https://github.com/cgpotts/cs224u

1.12

- https://gist.github.com/honnibal
- siamese lstm: https://github.com/aditya1503/Siamese-LSTM
- accuracy of lunar chinese calendar to predict baby sex http://onlinelibrary.wiley.com/doi/10.1111/j.1365-3016.2010.01129.x/abstract;
- customized keras lambda: https://gist.github.com/keunwoochoi

30.11

- rnn tricks: http://www.slideshare.net/indicods/general-sequence-learning-with-recurrent-neural-networks-for-next-ml
- data mining in action: Moscow, Russia: https://github.com/vkantor/MIPT_Data_Mining_In_Action_2016
- hypo testing, birthday effect: http://www.slideshare.net/SergeyIvanov105/birthday-effect-67829860
- LUI: linguistic UI https://medium.com/swlh/a-natural-language-user-interface-is-just-a-user-interface-4a6d898e9721
- fake news is 80% accuracy better: http://www.mallikarjunan.com/verytas/how-good-are-you-at-recognizing-satire-quiz
- nampi, spain 2017
- decode thought vector: http://gabgoh.github.io/ThoughtVectors/
- unstrained fmin: https://github.com/benfred/fmin
- neural programmer: https://github.com/tensorflow/models/tree/master/neural_programmer
- https://www.tensorflow.org/versions/master/how_tos/embedding_viz/index.html#tensorboard-embedding-visualization


29.11

- https://github.com/nyu-dl/NLP_DL_Lecture_Note
- NYU DL for NLP https://docs.google.com/document/d/1YS5QRvqMJVs9n3sK5fFjuldY7_vh42C5uUfxUGgL-Gc/edit
- http://tuanvannguyen.blogspot.com/2016/11/machine-learning-la-gi.html
- http://sebastianruder.com/cross-lingual-embeddings/
- https://docs.google.com/presentation/d/1O-Ics69y445aWuxQ_VW6SDvKT9BGl3ZXLLZDG9tUiUY/edit#slide=id.p

28.11

- event detection and deep learning: http://www.cs.nyu.edu/~thien/
- https://github.com/anoperson/NeuralNetworksForRE
- ED EE and MD with RNN and CNN: http://www.aclweb.org/anthology/P/P15/P15-2060.pdf

27.11

- http://www.slideshare.net/PyData/fang-xu-enriching-content-with-knowledge-base-by-search-keywords-and-wikidata
- https://www.mediawiki.org/wiki/Wikidata_query_service/User_Manual

26.11

- slides from mlconf sf 2016:http://www.slideshare.net/SessionsEvents/anjuli-kannan-software-engineer-google-at-mlconf-sf-2016
- http://www.slideshare.net/KenjiEsaki/kdd-2016-slide

25.11

- vo duy tin: https://github.com/duytinvo
- https://spacy.io/docs/usage/entity-recognition

24.11

- chinese NLP: https://github.com/taozhijiang/chinese_nlp
- not news: http://venturebeat.com/2016/11/23/twitter-cortex-team-loses-some-ai-researchers/
- sentihood: http://annotate-neighborhood.com/download/download.html, https://arxiv.org/pdf/1610.03771v1.pdf

23.11

Multithread in Theano:

- check your blas: https://raw.githubusercontent.com/Theano/Theano/master/theano/misc/check_blas.py
- http://deeplearning.net/software/theano/tutorial/multi_cores.html?highlight=multi%20co
- https://github.com/Theano/Theano/issues/3239
- set OMP_NUM_THREADS=4 inside the notebook with env: https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

Debug

- torch vs theano vs tf: https://www.quora.com/Is-TensorFlow-better-than-other-leading-libraries-such-as-Torch-Theano
- debug Deep Learning: https://gab41.lab41.org/some-tips-for-debugging-deep-learning-3f69e56ea134#.1ldbphlav
- negative loss: https://github.com/fchollet/keras/issues/1917

- CAP: Clustering Association Prediction, stas thinking https://www.researchgate.net/publication/310597778_Scientific_discovery_through_statistics

22.11

- stance detection: favour or against: http://isabelleaugenstein.github.io/papers/SemEval2016-Stance.pdf
- Hugo from Twitter to Google Brain, Montreal: https://techcrunch.com/2016/11/21/google-opens-new-ai-lab-and-invests-3-4m-in-montreal-based-ai-research/?sr_share=facebook
- train word2vec in gensim in good way: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

21.11

- sparql in python: https://joernhees.de/blog/tag/install/
- minhash: http://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/
- beating the kaggle easy way: http://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf

19.11

- 10 takeaways writeup MLConf SF: https://tryolabs.com/blog/2016/11/18/10-main-takeaways-from-mlconf/
- theano summer school: https://github.com/mila-udem/summerschool2015
- gpu card for macbook pro: http://udibr.github.io/using-external-gtx-980-with-macbook-pro.html
- transfer learning using pretrained vgg, resnet for your problem: https://github.com/dolaameng/transfer-learning-lab

18.11

- wikidata sparql: https://docs.google.com/presentation/d/16HhxRH-kkxqxcyzepXT-dHrnE90yVPlfkPq3cM2UzFg/edit#slide=id.g18e33c9ee6_2_134
- unkify: https://github.com/cdg720/emnlp2016/blob/master/utils.py#L322
- http://smerity.com/articles/2016/google_nmt_arch.html

17.11

- wikidata: http://www.slideshare.net/_Emw/an-ambitious-wikidata-tutorial
- wptools: https://github.com/siznax/wptools/wiki
- google translate: https://arxiv.org/pdf/1611.04558v1.pdf
- https://arxiv.org/pdf/1611.05104v1.pdf
- https://arxiv.org/pdf/1611.01587v2.pdf

16.11

- dssm deep sem sim models: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wsdm2015.v3.pdf
- twitter @ Singapore: http://www.straitstimes.com/singapore/twitter-eyes-local-talent-for-singapore-data-science-team
- multiple tasks of NLP: https://arxiv.org/pdf/1611.01587v2.pdf
- QUASI RNN: https://arxiv.org/pdf/1611.01576v1.pdf

15.11

- regex learning: http://dlacombejr.github.io/2016/11/13/deep-learning-for-regex.html
- recurrent + cnn for text classification: https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier
- quiver: to view convnet layer https://github.com/jakebian/quiver
- hera: to see training progress board: https://github.com/jakebian/hera
- RAISR: Rapid and Accurate Image Super Resolution https://arxiv.org/pdf/1606.01299v3.pdf
- why is machine learning hard: http://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html

14.11

- event ODSC West: https://www.odsc.com/california
- MLconf SF 12 Nov, summary: https://github.com/adarsh0806/ODSCWest/blob/master/MLConf.md
- Duy Do talk: https://speakerdeck.com/duydo/elasticsearch-for-data-engineers


13.11

- barcampsaigon 2016: some good topics on Elastic Search (Duy Do), Big Data analytics (Trieu Nguyen)
- Altair https://speakerdeck.com/jakevdp/visualization-in-python-with-altair

12.11

- Applications to explore (most of them are keras based)
- https://github.com/farizrahman4u/seq2seq
- https://github.com/farizrahman4u/qlearning4k
- https://github.com/matthiasplappert/keras-rl

- http://ml4a.github.io/guides/

- https://github.com/kylemcdonald/SmileCNN
- https://github.com/jocicmarko/ultrasound-nerve-segmentation
- https://github.com/abbypa/NNProject_DeepMask
- https://github.com/awentzonline/keras-rtst

- https://github.com/phreeza/keras-GAN
- https://github.com/jacobgil/keras-dcgan

- https://github.com/mokemokechicken/keras_npi
- https://github.com/codekansas/keras-language-modeling
    
11.11

- https://github.com/wiki-ai/revscoring
- Visual OCR attention: https://github.com/da03/Attention-OCR
- startup and DL: https://github.com/lipiji/App-DL
- embed + encode + attend + predict: https://explosion.ai/blog/deep-learning-formula-nlp
- HN: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

10.11

- https://arxiv.org/pdf/1508.06615.pdf


9.11

- ibm researcher, lda gib sampling, doc2vec: https://github.com/jhlau

8.11

- quoc le, rnn with reinforcement learning: http://openreview.net/pdf?id=r1Ue8Hcxg

7.11

- https://github.com/vinhkhuc/MemN2N-babi-python
- similarity proximity: http://www.datasciencecentral.com/profiles/blogs/comparison-between-global-vs-local-normalization-of-tweets-and
- pycon15, elastic search: https://github.com/erikrose/elasticsearch-tutorial

6.11

- https://github.com/Keats/rodent

04.11

- airbnb knowledge scale: https://medium.com/airbnb-engineering/scaling-knowledge-at-airbnb-875d73eff091#.5moos4eki
- R notebooks: http://rmarkdown.rstudio.com/r_notebooks.html
- dask: https://github.com/dask/dask
- dask vs celery: http://matthewrocklin.com/blog/work/2016/09/13/dask-and-celery
- dask in jupyperlab: https://learning.acm.org/webinar_pdfs/ChristineDoig_WebinarSlides.pdf

3.11

- https://hbr.org/resources/pdfs/hbr-articles/2016/11/the_state_of_machine_intelligence.pdf
- shallow learn: gensim + fasttext: https://github.com/giacbrd/ShallowLearn
- nn for sa: http://www.emnlp2016.net/tutorials/zhang-vo-t4.pdf

2.11

- mask bilstm: http://dirko.github.io/Bidirectional-LSTMs-with-Keras/
- https://github.com/clab/dynet_tutorial_examples
- data scientist at Facebook: https://tctechcrunch2011.files.wordpress.com/2016/11/data-scientist.png?w=80&h=60&crop=1
- reality vs expectation: https://pbs.twimg.com/media/CwNNnXfXEAIT-kY.jpg
- swagger + flask: http://michal.karzynski.pl/blog/2016/06/19/building-beautiful-restful-apis-using-flask-swagger-ui-flask-restplus/
- godfather of vis: https://pbs.twimg.com/media/CwNEBlsXYAAOm8G.jpg

1.11

- https://github.com/nhwhite212/Dealing-with-Data-Spring2016
- mongodb: http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
- https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment
- decompositional attetion: https://arxiv.org/pdf/1606.01933v2.pdf

31.10

- attention lstm: http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
- memory and attention, pointer: http://slides.com/smerity/quora-frontiers-of-memory-and-attention#/75

30.10

- https://www.oreilly.com/ideas/what-is-hardcore-data-science-in-practice?platform=hootsuite
- hybrid neural networks with memory: alex grave, Nature.

29.10

- multi-tag doc2vec: https://github.com/sindbach/doc2vec_pymongo
- https://dineshrai.herokuapp.com/posts/how-i-used-scikit-learn-to-find-good-eyes

28.10

- nyu dl4mt: https://github.com/nyu-dl/dl4mt-tutorial
- stock vis, phvu: https://github.com/phvu/misc/tree/master/stock_visualizer
- https://github.com/blue-yonder/tsfresh


27.10

- gensim in Russian at Yandex: https://www.youtube.com/watch?v=U0LOSHY7U5Q

26.10

- how to make a data vis: https://drive.google.com/drive/folders/0BxYkKyLxfsNVd0xicUVDS1dIS0k
- cntk: https://t.co/lMdjVfTKgE
- torch fast softmax: https://code.facebook.com/posts/1827693967466780/
- lua, rnn generates clickbait https://github.com/larspars/word-rnn
- https://github.com/sindbach/doc2vec_pymongo
- cntk rank 7: https://github.com/Microsoft/CNTK

25.10

- google brain, reading, notes, papers of Denny: https://github.com/dennybritz/cnn-text-classification-tf
- http://www.emnlp2016.net/tutorials/chen-liu-t3.pdf
- feature engineering: https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering/answer/Claudia-Perlich?srid=cgo
- https://github.com/uclmr/stat-nlp-book
- http://varianceexplained.org/r/bayesian-ab-testing/
- to try GA in R: https://github.com/michalbrys/R-Google-Analytics

24.10

- unsup langid: http://blog.echen.me/2011/05/01/unsupervised-language-detection-algorithms/
- https://github.com/budhiraja/languageIdentification/blob/master/Identification%20of%20Language.ipynb

23.10

- xss detection: https://github.com/faizann24/XssPy
- bad url: https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
- pentest: https://github.com/faizann24/Resources-for-learning-Hacking

22.10

- https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap

21.10

- longitudinal analysis: http://www.cs.odu.edu/~anwala/files/temp/archivesUnleashedHackathon/Hebdo_Twitter.html
- tracking discourse event on twitter: https://docs.google.com/presentation/d/1Q6vZdLbOS98kRLQv3hPQeXj77jO-PGdPN_CzupWOYGA/edit#slide=id.p
- mac archey: http://www.mitchchn.me/2014/os-x-terminal/
- google kb: https://developers.google.com/knowledge-graph/


20.10

- http://blog.mldb.ai/blog/posts/2016/10/deepteach/
- http://www.phontron.com/slides/bayesian-nonparametrics.pdf
- beamsearch: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/46927-f97/slides/Lec3/sld023.htm
- beamsearch in skipthough sequence generated: https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
- https://blog.heuritech.com/2015/12/01/learning-to-link-images-with-their-descriptions/
- https://github.com/proycon/pynlpl
- spacy 1.0 + keras: https://explosion.ai/blog/spacy-deep-learning-keras
- fipi: Automatized political analysis of text http://fipi-dev.elasticbeanstalk.com/news.html
- fipi github: https://github.com/felixbiessmann/fipi

18.10

- http://www.bbc.com/news/technology-37684418
- https://github.com/ogencoglu/Templates/tree/master/Python
- progressbat tqdm: https://pypi.python.org/pypi/tqdm

17.10

- text to image: thought vector: https://github.com/paarthneekhara/text-to-image

16.10

- https://github.com/Dyakonov/case_sdsj
- https://nbviewer.jupyter.org/github/Dyakonov/case_sdsj/blob/master/dj_sdsj01_visual.ipynb
- https://alexanderdyakonov.wordpress.com/2015/11/06/%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%BC%D1%81%D1%82%D0%B2%D0%BE-%D1%81-pandas-%D1%81%D0%BB%D0%B0%D0%B9%D0%B4%D1%8B/
- http://blog.xukui.cn/
- https://github.com/uclmr/emoji2vec
- https://danielmiessler.com/blog/machine-learning-new-statistics/#gs.gpTYESc
- https://github.com/fchollet/keras/issues/1400

15.10

- https://www.scripted.com/content-marketing-2/36-years-presidential-debates

14.10

- SemEval: http://www.saifmohammad.com/WebPages/StanceDataset.htm
- stance detection, implicitly: https://github.com/sheffieldnlp/stance-conditional
- https://github.com/transcranial/keras-js
- http://www.slideshare.net/isabelleaugenstein/weakly-supervised-machine-reading

13.10

- deep sequence classification: https://github.com/napsternxg/DeepSequenceClassification
- German NER, SENNA: https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2014/2014_GermEval_Nested_Named_Entity_Recognition_with_Neural_Networks.pdf
- fin keyword expansion: tf + tfidf + log(1+tc): https://www.aclweb.org/anthology/D/D14/D14-1152.pdf

12.10

- deep font: https://erikbern.com/2016/01/21/analyzing-50k-fonts-using-deep-neural-networks/
- David talk: https://github.com/dav009/topictalk/blob/master/slides.pdf

11.10

- 8 cat companies classified: https://davidgrosfeld.files.wordpress.com/2010/03/revenuevisualization1.png
- pandas multilevel columns: http://stackoverflow.com/questions/21443963/pandas-multilevel-column-names
- https://github.com/plouismarie/basic-sigmoid-ML

10.10

- https://github.com/ewulczyn/wiki-misc
- https://github.com/vered1986/linker?files=1
- energy consumption prediction: http://cs229.stanford.edu/proj2015/345_report.pdf
- DL stock volatility and goog trend domestic: http://cs229.stanford.edu/proj2015/186_report.pdf
- learning and mining: https://github.com/shuyo/iir
- langid, cldr, ldig: https://github.com/shuyo/ldig
- langid for tweet: http://www.aclweb.org/anthology/W14-1303 
- http://www.dialog-21.ru/en/dialogue2016/results/program/day-1/
- nerc in russian: http://www.dialog-21.ru/media/3433/sysoevaaandrianovia.pdf

7.10

- wiki2vec: https://github.com/deliarusu/wikipedia-correlation/blob/master/wikipedia-correlation-ftse100.ipynb
- https://github.com/nadbordrozd/blog_stuff/tree/master/classification_w2v
- http://blog.datafox.com/the-data-driven-approach-to-finding-similar-companies/
- http://blog.ventureradar.com/2016/07/27/new-features-enhanced-similar-companies-lists-and-suggested-keywords/
- http://nadbordrozd.github.io/blog/2016/06/13/deepwalking-with-companies/

6.10

- resume job matching: https://arxiv.org/pdf/1607.07657.pdf
- https://github.com/lyoshiwo/resume_job_matching
- short text sim with word embedding: https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/kenter-short-2015.pdf
- yandex deeplearning school: https://github.com/yandexdataschool/HSE_deeplearning
- http://haixun.olidu.com/Probase.Feb.2014.pdf

5.10

- turin nlp4twitter: https://github.com/leoferres/nlp4twitter_tutorial
- spacy nlp: https://github.com/NSchrading/intro-spacy-nlp/blob/master/Intro_spaCy_NLP.ipynb
- twitter calendar: http://ec2-54-170-89-29.eu-west-1.compute.amazonaws.com:8000/
- https://maxdemarzi.com/2012/08/10/summarize-opinions-with-a-graph-part-1/
- http://aclanthology.info/
- https://github.com/aritter/twitter_nlp
- https://github.com/deliarusu/pydata_berlin2016_materials

3.10

- https://github.com/ijmbarr/panama-paper-network
- https://maxdemarzi.com/2012/08/10/summarize-opinions-with-a-graph-part-1/
- http://www.slideshare.net/neo4j/natural-language-processing-with-graphs
- why clt: http://www.kdnuggets.com/2016/08/central-limit-theorem-data-science-part-2.html
- wikitext: http://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/
- auto headline from description: https://github.com/udibr/headlines?files=1
- https://handong1587.github.io/deep_learning/2015/10/09/nlp.html#summarization
- movies for ds: https://ongxuanhong.wordpress.com/2016/09/04/data-scientists-nen-xem-gi/
- opinosis: http://kavita-ganesan.com/opinosis

30.9

- nlp4fin: http://www3.nd.edu/~mcdonald/Word_Lists.html
- ssix: social sentiement index: http://ssix-project.eu/
- http://www3.nd.edu/~mcdonald/

29.9

- stats nlp book: https://github.com/uclmr/stat-nlp-book
- rep learning on twitter: https://github.com/cedricdeboom/RepresentationLearning
- short text rep: https://arxiv.org/pdf/1607.00570.pdf
- sa on post debation: https://www.datazar.com/project/p7dc15a26-4551-4c79-8e41-fbc7b84641fd
- https://blog.datazar.com/first-debate-2016-sentimental-analysis-of-candidates-58d87092fc6a#.iz6sgezbx

28.9

- r blogger: http://juliasilge.com/blog/Song-Lyrics-Across/
- fastText as hybrid recommender system: https://blog.lateral.io/2016/09/fasttext-based-hybrid-recommender/
- pointer sentinel mixture models: http://arxiv.org/pdf/1609.07843v1.pdf


27.9

- dl4nlp: http://web.eecs.umich.edu/~radev/intronlp/
- CommAI: https://github.com/facebookresearch/CommAI-env
- a roadmap towards ML: http://arxiv.org/pdf/1511.08130v2.pdf
- apply dl, nuts and bolts: https://kevinzakka.github.io/2016/09/26/applying-deep-learning/
- vis nlp: https://arxiv.org/pdf/1506.01066v2.pdf
- vis nn: https://civisanalytics.com/blog/data-science/2016/09/22/neural-network-visualization/

26.9

- LIWC: little words in big data: https://github.com/scottofthescience/liwcExtractor
- LIWC paper 2010, http://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2010.pdf
- TweetNLP: http://www.cs.cmu.edu/~ark/TweetNLP/
- acl 15: http://aclweb.org/anthology/P/P15/
- AKE: automatic keywords extraction for twitter: http://aclweb.org/anthology/P/P15/P15-2105.pdf
- lex comparison btw Wikipedia and Twitter corpus: http://aclweb.org/anthology/P/P15/P15-2108.pdf

25.9

- Y Lecun: DL and AI future: https://www.youtube.com/watch?v=wofXCQXq1pg&feature=youtu.be
- youtube rec: https://static.googleusercontent.com/media/research.google.com/vi//pubs/archive/45530.pdf
- lie detector: https://docs.google.com/presentation/d/1tHt5EPol3KLu81E8aKlebZT3FH0uvPky4l3bHEwVbEQ/mobilepresent?slide=id.g162ec24d83_0_0

23.9

- adam opt = sgd + momentum http://cs231n.github.io/neural-networks-3/#ada
- need4tweet: https://github.com/badiehm/TwitterNEED
- https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html
- http://colinmorris.github.io/blog/1b-words-char-embeddings

22.9

- http://www.kdnuggets.com/2015/03/deep-learning-text-understanding-from-scratch.html
- http://www.john-foreman.com/blog
- stats: http://andrewgelman.com/
- https://github.com/ujjwalkarn/DataSciencePython
- https://wwbp.org/blog/do-the-presidential-candidates-have-a-plan-or-highlight-problems/#more-408
- http://wwbp.org/blog/insights-to-the-2016-election/

21.9

- hack password with 3LSTM + 2 Dense: https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_melicher.pdf
- useful R wikipediatrend: https://github.com/petermeissner/wikipediatrend

20.9

- http://deadtreepages.com/posts/python-code-character-prediction-with-lstm-rnn.html
- why is your data valuable? https://github.com/ogencoglu/MyData2016
- http://www.cs.tut.fi/kurssit/SGN-41006/slides/
- fast region-cnn: https://github.com/rbgirshick/py-faster-rcnn
- automatic text scoring: https://github.com/dimalik/ats, http://www.aclweb.org/anthology/P/P16/P16-1068.pdf
- https://www.zybuluo.com/HaomingJiang/note/462804
- model extraction attack: https://github.com/ftramer/Steal-ML
- nlp and dl from India: https://github.com/rishy?tab=repositories
- http://slides.com/rishabhshukla/deep-learning-in-nlp#/
- https://github.com/hungtraan/FacebookBot

19.9

- https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
- Cortex team, RL: https://blog.twitter.com/2016/reinforcement-learning-for-torch-introducing-torch-twrl

15.9

- altair: http://nbviewer.jupyter.org/github/ellisonbg/altair/blob/master/altair/notebooks/12-Measles.ipynb
- nice illustration: http://www.asimovinstitute.org/neural-network-zoo/
- https://github.com/shashankg7/Deep-Learning-for-NLP-Resources
- https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Word2Vec_FastText_Comparison.ipynb

14.9

- vqa: https://github.com/iamaaditya/VQA_Demo/blob/master/demo.py
- wavenet in keras: https://github.com/usernaamee/keras-wavenet
- why jupyter notebook so cool: http://arogozhnikov.github.io/2016/09/10/jupyter-features.html
- fin news aggregator: https://finzine.com/

13.9

- thought vector, Hinton again: http://deeplearning4j.org/thoughtvectors
- semantic parsing, extracting B-dep, B-arr, B-date from airline info: http://deeplearning.net/tutorial/rnnslu.html
- deep learning intro in Taiwan: http://mail.tku.edu.tw/myday/teaching/1042/SCBDA/1042SCBDA09_Social_Computing_and_Big_Data_Analytics.pdf
- tankbuster: https://github.com/thiippal/tankbuster/blob/master/README.md
- trending arxiv: https://github.com/Smerity/trending_arxiv

9.9

- pydataberlin 2016: https://github.com/deeplook/pydata_berlin2016_materials
- wavenet: https://deepmind.com/blog/wavenet-generative-model-raw-audio/

8.9

- use vgg for image classification: https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
- guide to phd from Karpathy: http://karpathy.github.io/2016/09/07/phd/
- image captioning, thesis of Karpathy: http://cs.stanford.edu/people/karpathy/main.pdf
- step data in Iphone: http://blog.yhat.com/posts/phone-steps-timeseries.html

7.9

- movidius VPU: http://www.forbes.com/sites/aarontilley/2016/01/27/google-partners-with-chip-startup-to-take-machine-learning-out-of-the-cloud-and-into-your-pocket/?linkId=20717947#6c21345a33e5
- google news 3M to 300K, slim version: https://github.com/eyaler/word2vec-slim


6.9

- source code classification: http://blog.aylien.com/source-code-classification-using-deep-learning/
- aaqi ml for text, sound: https://github.com/aqibsaeed
- data science at Google: http://www.unofficialgoogledatascience.com/2015/11/how-to-get-job-at-google-as-data.html
- is your model stat significantly improved? https://www.reddit.com/r/MachineLearning/comments/519z41/when_would_i_say_that_my_classifier_performance/

5.9

- learn to learn: http://nuit-blanche.blogspot.com/2016/09/saturday-morning-videos-learning-to.html
- big data and stochastic algos: http://www.proba.jussieu.fr/SlidesAlgoSto/AlgoStoch_Gallinari.pdf
- random bits on data: https://datathinking.wordpress.com/author/dolaameng/
- hitchhiker guide to the galaxy: http://izt.ciens.ucv.ve/ecologia/Archivos/Filosofia-II/Adams,%20Douglas%20-%20The%20Hitchhikers%20Guide%20To%20The%20Galaxy.pdf
- deep learning 42 https://www.youtube.com/watch?v=furfdqtdAvc&feature=youtu.be
- optimization algos in Haskell
- learn to learn and decompositionality with deep nn: https://www.youtube.com/watch?v=x1kf4Zojtb0

2.9

- sum product networks: https://arxiv.org/abs/1608.08266
- keras application, used inception v3: https://github.com/danielvarga/keras-finetuning
- keras: neural style: https://github.com/titu1994/Neural-Style-Transfer

1.9

- pseudo topic modeling PTM: http://www.kdd.org/kdd2016/subtopic/view/topic-modeling-of-short-texts-a-pseudo-document-view
- cucumber classification: https://cloud.google.com/blog/big-data/2016/08/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow
- implemented bigru on tweet2vec: https://arxiv.org/pdf/1605.03481v2.pdf
- deep learning for computing vision, Barcelona, Spain 2016: http://imatge-upc.github.io/telecombcn-2016-dlcv/

31.8

- doc2text: https://github.com/jlsutherland/doc2text
- synthetic gradient: https://deepmind.com/blog#decoupled-neural-interfaces-using-synthetic-gradients
- Oriol vanyals, pioneer in s2s got 35 under 35 award: https://www.technologyreview.com/lists/innovators-under-35/2016/pioneer/oriol-vinyals/
- https://github.com/baidu/paddle

29.8

- text to image: https://github.com/paarthneekhara/text-to-image/blob/master/skipthoughts.py
- forex prediction UP, DOWN, MID: https://is.muni.cz/th/422802/fi_b/bakalarka_final.pdf

28.8

- keras + lasagne: https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
- last week: pydatachi2016, pyconmy2016
- https://github.com/y3l2n/pydata2016-notes
- sebastian raschka, ml tut: https://github.com/rasbt/pydata-chicago2016-ml-tutorial/blob/master/code/tutorial.ipynb

26.8

- euroscipy with kera: https://github.com/leriomaggio/deep-learning-keras-euroscipy2016/
- deep mask from Facebook


25.8

- lstm vis: http://lstm.seas.harvard.edu/
- ML at Apple: https://backchannel.com/an-exclusive-look-at-how-ai-and-machine-learning-work-at-apple-8dbfb131932b#.asw5moh3s
- https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html
- https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/


24.8

- http://benjaminbolte.com/blog/2016/keras-language-modeling.html
- https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
- http://swanintelligence.com/first-steps-with-neural-nets-in-keras.html
- https://groups.google.com/forum/#!topic/keras-users/9C5emrJRRUs
 
23.8

- https://www.aclweb.org/aclwiki/index.php?title=State_of_the_art
- http://jmlr.org/proceedings/papers/v48/johnson16.pdf

22.8

- https://webcourse.cs.technion.ac.il/232601/Spring2016/en/ho.html
- http://www.deeplearningbook.org/contents/guidelines.html
- http://mickypaganini.github.io/

19.8

- sensity analysis for sentence classification: http://arxiv.org/pdf/1510.03820v4.pdf
- https://code.facebook.com/posts/1438652669495149/fair-open-sources-fasttext
- https://github.com/dolaameng/deeplearning-exploration

18.8

- https://www.springboard.com/blog/data-science-interviews-lessons/
- https://medium.com/autonomous-agents/how-to-train-your-neuralnetwork-for-wine-tasting-1b49e0adff3a#.5dzpfqoyz


16.8

- http://www.wangzhongyuan.com/tutorial/ACL2016/Understanding-Short-Texts/
- make your pip: http://marthall.github.io/blog/how-to-package-a-python-app/

15.8

- https://github.com/poliglot/fasttext/blob/master/README.md
- fasttext python: https://pypi.python.org/pypi/fasttext/0.6.1

14.8

- https://github.com/dgrtwo/JSM2016slides
- http://nlpers.blogspot.com/2016/08/fast-easy-baseline-text-categorization.html
- https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py

13.8

- sequence to poetry: http://arxiv.org/pdf/1511.06349.pdf
- http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/
- http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf
- http://techcitynews.com/2016/08/11/tech-giants-embracing-artificial-intelligence/
- https://code.facebook.com/posts/181565595577955/introducing-deeptext-facebook-s-text-understanding-engine/

12.8

- sentiment analysis, alway good things from Chris Pott: http://sentiment.christopherpotts.net/index.html
- wwbp: https://penncurrent.upenn.edu/2016-03-24/latest-news/penn-scholars-distinguish-presidential-candidates-analyzing-their-words/

10.8

- http://varianceexplained.org/r/trump-tweets/
- http://koaning.io/theme/iframes/ams-meetup-notebook.html


9.8

- short text understanding http://sci-hub.cc/10.1109/tkde.2015.2485224
- https://github.com/johnymontana/nlp-graph-notebooks/blob/master/opinion_mining.ipynb
- world well being project http://wwbp.org/blog/
- fastText benchmark: https://github.com/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb
- wp bot: https://techcrunch.com/2016/08/05/robots-will-cover-the-olympics-for-the-washington-post/
- http://datascientistjobinterview.com/
- https://hackerlists.com/tensorflow-resources/


8.8

- twitter buy magic pony: $150M, apple buy Turi $200: time of ML
- spotify release rader: brand new music from acoustic frame


7.8

- https://sites.google.com/site/acl16nmt/
- roc curve vs pr curve: http://numerical.recipes/CS395T/lectures2008/17-ROCPrecisionRecall.pdf
- https://ru.coursera.org/learn/vvedenie-mashinnoe-obuchenie/lecture/P9Zun/mnoghoklassovaia-klassifikatsiia
- hpv: https://github.com/lukaselmer/hierarchical-paragraph-vectors
- nlp pointers: https://gist.github.com/mattb/3888345
- vowpal wabbit: https://github.com/hal3/vwnlp

5.8

- I am a data scientist: https://yanirseroussi.com/2016/08/04/is-data-scientist-a-useless-job-title/
- Generative modeling, OpenAI, Ilya Sutskever: http://scaledml.org/2016/slides/ilya.pdf
- HOT: fasttext from FB: https://github.com/facebookresearch/fastText
- OMG, data leak, how to have trick on ID, time series: https://alexanderdyakonov.wordpress.com/2016/07/27/id-и-время/

4.8

- tweep in tsne: https://github.com/abhshkdz/tsne-top-indian-tweeps
- dl is not a hamer: I try ngram + tfidf to beat cnn on AG corpus: http://arxiv.org/pdf/1509.01626v3.pdf
- fasttext in torch: https://github.com/kemaswill/fasttext_torch
- super cool: self driving with open source http://research.comma.ai/
- coreNLP: http://www.lewisgavin.co.uk/NLP/

3.8

- categorize tumblr post: https://engineering.tumblr.com/post/148350944656/categorizing-posts-on-tumblr
- https://github.com/jayantj/w2vec-similarity
- https://jayantj.github.io/posts/project-gutenberg-word2vec
- reuter on keras + gensim: http://www.bonaccorso.eu/2016/08/02/reuters-21578-text-classification-with-gensim-and-keras/

2.8

- trendminer: http://www.trendminer-project.eu/index.php/downloads
- https://sites.sas.upenn.edu/danielpr/pages/resources
- https://github.com/danielpreotiuc/textrank
- text rank: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
- hierarchy doc2vec: http://e-collection.library.ethz.ch/eserv/eth:48203/eth-48203-01.pdf
- probase roadmap: http://tcci.ccf.org.cn/summersch/classm/ADL32-Lecture03-Report3.pdf
- understanding short text: http://www.wangzhongyuan.com/tutorial/ACL2016/Understanding-Short-Texts/
- http://tech.marksblogg.com/airflow-postgres-redis-forex.html
- keras: text + image --> http://cbonnett.github.io/Insight.html
- lstm viz: https://arxiv.org/pdf/1506.01066v2.pdf

1.8

- hacker math for DS: https://github.com/amitkaps/hackermath
- word2vec from theo to prac: http://hen-drik.de/pub/Heuer%20-%20word2vec%20-%20From%20theory%20to%20practice.pdf
- twitter job/occupation: http://www.lampos.net/sites/default/files/papers/twitterJobs_ACL15.pdf
- https://www.clarifai.com/#demo
- word dominant: http://compling.hss.ntu.edu.sg/courses/hg7017/pdf/word2vec%20and%20its%20application%20to%20wsd.pdf
- word2vec, tsne, d3js: https://github.com/h10r/topic_comparison_tool
- organic tweet: http://arxiv.org/pdf/1505.04342v6.pdf

28.7

- https://ireneli.eu/2016/05/17/nlp-04-an-log-linear-model-for-tagging-task-python/
- lexvec: https://github.com/alexandres/lexvec

27.7

- yandex data school (some in Russian): https://github.com/yandexdataschool/mlhep2016
- https://github.com/vladsandulescu/topics
- https://github.com/vladsandulescu/phrases

26.7

- caffe: http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/
- neo4j: https://github.com/johnymontana/graph-of-thrones/blob/master/network-of-thrones.ipynb
- doc classification with DL: http://home.iitk.ac.in/~amlan/cs671/project/slides.pdf
- https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
- beyond cosine: http://stefansavev.com/blog/beyond-cosine-similarity/
- disco twitter: http://stanford.edu/~rezab/papers/disco.pdf
- dl glossary: http://www.wildml.com/deep-learning-glossary/
- http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
- textacy: https://github.com/bdewilde/pygotham_2016/blob/master/pygotham_2016.pdf


25.7

- https://github.com/rouseguy/europython2016_dl-nlp


24.7

- http://stefansavev.com/blog/beyond-cosine-similarity/
- http://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

22.7

- http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/


21.7

- clustering measurement: https://www.youtube.com/watch?v=Mf6MqIS2ql4, http://www.slideshare.net/PyData/christian-henning-assessing-the-quality-of-a-clustering
- pandas tut: https://github.com/jonathanrocher/pandas_tutorial
- hdbscan: https://www.youtube.com/watch?v=AgPQ76RIi6A&index=19&list=PLYx7XA2nY5Gf37zYZMw6OqGFRPjB1jCy6
- python app, flask --> flexx: https://www.youtube.com/watch?v=kIPL3N2Xk_c&index=22&list=PLYx7XA2nY5Gf37zYZMw6OqGFRPjB1jCy6
- pcap, feature engineering, sklearn for security: https://www.youtube.com/watch?list=PLYx7XA2nY5Gf37zYZMw6OqGFRPjB1jCy6&v=0KXfRGD-Ins
- sentiment https://github.com/lab41/sunny-side-up
- radim: http://rare-technologies.com/sigir2016_169.pdf

20.7

- can we predict https://gab41.lab41.org/can-word-vectors-help-predict-whether-your-chinese-tweet-gets-censored-711e7682d12f#.n0670sw5j
- dynamicCNN: https://github.com/FredericGodin/DynamicCNN
- SIGIR2016: http://nlp.stanford.edu/~manning/talks/SIGIR2016-Deep-Learning-NLI.pdf
- embedding metrics: https://github.com/julianser/hed-dlg-truncated/blob/master/Evaluation/embedding_metrics.py

19.7

- design and experiment https://github.com/juanshishido/experiments-guide
- fluent python notebook: https://github.com/juanshishido/fluent-python-notebooks
- yahoo text categorization: https://github.com/juanshishido/text-classification
- OKCupid: https://www.youtube.com/results?sp=SBTqAwA%253D&q=scipy+2016
- https://github.com/juanshishido/okcupid
- topic coherence: http://rare-technologies.com/validating-gensims-topic-coherence-pipeline/
- weighted random generation: http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
- nlp to ir: http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00239
- nlu icml 2015:http://icml.cc/2015/tutorials/icml2015-nlu-tutorial.pdf
- bayesian time series: http://icml.cc/2015/tutorials/BayesianTimeSeries.pdf


18.7

- tf on skflow: https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.x1s9r6nmh
- plot with point: http://walkerke.github.io/geog30323/slides/eda-2/#/20
- Bayesian Poisson Tensor Factorization: http://arxiv.org/abs/1506.03493
- dirichlet process mixture model: https://github.com/hannawallach/dpmm
- scikit tensor: https://github.com/mnick/scikit-tensor
- bptf: https://github.com/aschein/bptf
- icml 2015 tutorial: http://videolectures.net/icml2015_lille/

15.7

- jupyter lab: https://github.com/jupyter/jupyterlab
- intro image cnn: https://github.com/rouseguy/scipyUS2016_dl-image/tree/master/notebooks
- intro nlp rnn: https://github.com/rouseguy/intro2deeplearning/tree/master/notebooks
- https://speakerdeck.com/chdoig/scaling-ds-in-python
- http://blog.jupyter.org/2016/07/14/jupyter-lab-alpha/?utm_content=buffere6cf2&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

14.7

data science summit:

- http://conf.turi.com/2016/us/agenda_day2/
- http://www.kdnuggets.com/2016/07/data-science-summit-2016-9-talks.html
- http://multithreaded.stitchfix.com/blog/2016/07/13/conf-talks-summer2016/

daily

- supervised algo empirical study: http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf
- data driven modeling: http://jakehofman.com/ddm/2012/02/lecture-03-2/
- word embedding for fun and profit: https://speakerdeck.com/tmylk/word-embeddings-for-fun-and-profit-with-gensim-pydata-london-2016
- #scipy2016: https://speakerdeck.com/astrofrog/python-3-for-scientists
- @DataSciSummit: 12, 13 July San Francisco: http://conf.turi.com
- https://github.com/chdoig/dss-scaling-tutorial
- pokemon location in spreadsheet: https://docs.google.com/spreadsheets/d/1G8U44ukuPdM9OfpHZt3kiUYYPrnxaRccLMXvxvEvMiI/edit#gid=869820177
- nbflow: https://github.com/jhamrick/nbflow
- http://www.jeannicholashould.com/the-theorem-every-data-scientist-should-know-2.html
- tweet2news: https://sites.google.com/site/engmathstwitternews/home
- meetup document classification: http://www.meetup.com/Big-Data-Israel/events/232569748/
- build tool: http://scons.org/
- tweet lda to aspect: http://sci-hub.cc/10.1007/978-3-642-37401-2_35
- tensorflow: deep and wide: https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html
- http://opensource.datacratic.com/mtlpy50/
- http://colah.github.io/posts/2015-01-Visualizing-Representations/

13.7

- tf practice: https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html
- wide and deep together: https://research.googleblog.com/
- sklearn + dask: http://jcrist.github.io/dask-sklearn-part-1.html
- image: dl framework: https://pbs.twimg.com/media/ClAUr5EUkAA9--0.jpg:large
- image: knowledge: https://pbs.twimg.com/media/CifCYeSUUAArriB.jpg:large
- numpy, scipy and pandas way to go: https://plot.ly/~empet/13902/numpy-cluster-in-the-network-of-python-packages/
- rnn live stream, character image: https://www.youtube.com/watch?v=wSpPJtenw_c
- big data 5cent https://pbs.twimg.com/media/CnMAV2XXgAAkBJY.jpg
- google research: https://plus.google.com/+ResearchatGoogle/posts
- deep learning word cloud: https://pbs.twimg.com/media/CnLJLdVXYAEpN6w.jpg:large
- jupyter on rpi: http://makeyourownneuralnetwork.blogspot.de/2016/03/ipython-neural-networks-on-raspberry-pi.html
- 

12.7

- http://mghassem.mit.edu/insights-word2vec/
- home depot: search relevance: https://github.com/ChenglongChen/Kaggle_HomeDepot/blob/master/Doc/Kaggle_HomeDepot_Turing_Test.pdf
- sentiment http://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
- reading tea leaves: https://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf
- https://www.linkedin.com/pulse/putting-semantic-representational-models-test-tf-idf-k-means-parsa?forceNoSplash=true


11.7

- gensim phrase: http://www.markhneedham.com/blog/2015/02/12/pythongensim-creating-bigrams-over-how-i-met-your-mother-transcripts/
- xgboost out of the box: https://www.quora.com/What-machine-learning-approaches-have-won-most-Kaggle-competitions/answer/Ben-Hamner?srid=cgo&share=45a4f6de
- ES part 2: http://insightdataengineering.com/blog/elasticsearch-core/
- GAN: https://www.youtube.com/watch?v=deyOX6Mt_As

8.7

- lsa + classification: https://github.com/chrisjmccormick/LSA_Classification
- perturbation + adversarial lstm https://arxiv.org/pdf/1605.07725v1.pdf
- I don't drink tiger (beer) confused neural transalation lisa: http://104.131.78.120/
- always love Mikolov related: fastText https://arxiv.org/abs/1607.01759
- Explaining the classisfier: https://www.youtube.com/watch?v=hUnRCxnydCc
- LIME: https://github.com/marcotcr/lime
- https://artistdetective.wordpress.com/2016/06/15/how-to-teach-a-computer-common-sense/
- 6 cons, top uni + org: http://www.marekrei.com/blog/analysing-nlp-publication-patterns/
- https://github.com/ijmbarr/panama-paper-network/blob/master/panama_network.ipynb

7.7

- https://engineers.sg/conference/pyconsg2016
- https://speakerdeck.com/tmylk/americas-next-topic-model?slide=6
- http://aclweb.org/anthology/J93-1003


6.7

- machine learning done wrong: http://dataskeptic.com/epnotes/machine-learning-done-wrong.php
- https://archive.org/details/twitterstream
- https://github.com/lintool/twitter-tools
- CLT: http://www.jeannicholashould.com/the-theorem-every-data-scientist-should-know.html
- https://blog.init.ai/three-impactful-machine-learning-topics-at-icml-2016-465be5ae63a#.yxw5wiisw
- http://www.machinedlearnings.com/2016/07/icml-2016-thoughts.html?spref=tw&m=1

5.7

- w2v writeup: https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/

4.7

- bot: http://52bots.tumblr.com/post/108322694954/11-ebook-of-black-earth-what-an-ebooks-style
- http://www.degeneratestate.org/posts/2016/Apr/20/heavy-metal-and-natural-language-processing-part-1/
- http://aclweb.org/anthology/J93-1003
- https://www.thefinancialist.com/man-vs-machine-what-happens-when-machines-can-learn-2/
- https://twimlai.com/fatal-ai-autopilot-crash-eu-may-prohibit-machine-learning-twiml-20160701/
- https://www.coursera.org/learn/natural-language-processing

1.7

- word2vec pipeline: https://github.com/NIHOPA/pipeline_word2vec
- visual recognition: http://cs231n.github.io/
- chris olah cv: https://colah.github.io/cv.pdf
- Google ML tut: https://www.youtube.com/watch?v=cSKfRcEDGUs
- ES: anatomy: http://insightdataengineering.com/blog/elasticsearch-crud/
- https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html
- http://arxiv.org/pdf/1512.00567v3.pdf

30.6

- voice detection: wav --> features --> SVM + RF + XGB --> RF --> prediction: http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/
- wide and deep: https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn

29.6

- hyperword: https://bitbucket.org/omerlevy/hyperwords
- wmd 20newsgroup: http://vene.ro/blog/word-movers-distance-in-python.html
- init orthogonal mat for RNN: http://smerity.com/articles/2016/orthogonal_init.html
- markov chain: https://github.com/fatiherikli/markov-chain-demo
- https://medium.com/@bjacobso/is-it-brunch-time-ffe3adf485d8#.sx662tn0x
- singapore statup eco: https://docs.google.com/document/d/1AsSwH_kJ5qm7X8Pb5H2P_iaCpUA9tr-93GGNuN-orXk/mobilebasic?pli=1
- http://www.datameer.com/company/datameer-blog/big-data-ecosystem/
- https://www.facebook.com/notes/bui-hai-an/v%C3%A0i-l%E1%BB%9Di-khuy%C3%AAn-thu-nh%E1%BA%B7t-%C4%91%C6%B0%E1%BB%A3c-t%E1%BB%AB-ges-p2/10153489287901106
- https://www.facebook.com/notes/bui-hai-an/v%C3%A0i-l%E1%BB%9Di-khuy%C3%AAn-thu-nh%E1%BA%B7t-%C4%91%C6%B0%E1%BB%A3c-t%E1%BB%AB-ges-p1/10153478669101106?notif_t=like&notif_id=1467086253182050

28.6

- http://lstm.seas.harvard.edu/
- https://www.reddit.com/r/MachineLearning/comments/4q5fsu/advanced_word_embeddings_for_seq2seq_applications/
- https://www.dataquest.io/blog/data-science-newsletters/
- http://nbviewer.jupyter.org/github/taddylab/deepir/blob/master/w2v-inversion.ipynb
- http://www.pyimagesearch.com/2016/06/27/my-top-9-favorite-python-deep-learning-libraries/

27.6

- document classification: https://github.com/RaRe-Technologies/movie-plots-by-genre
- classical nlp: https://github.com/tmylk/pycon-2016-nlp-tutorial/blob/master/jupyter/classical-nlp/classical-nlp.ipynb
- document classification: https://speakerdeck.com/tmylk/document-classification-with-word2vec-at-pydata-nyc
- inverse word2vec with hs: http://nbviewer.jupyter.org/github/taddylab/deepir/blob/master/w2v-inversion.ipynb
- wmd: http://tech.opentable.com/2015/08/11/navigating-themes-in-restaurant-reviews-with-word-movers-distance/
- defense of w2v: http://www.cs.tau.ac.il/~wolf/papers/qagg.pdf
- plagiarism: http://douglasduhaime.com/blog/cross-lingual-plagiarism-detection-with-scikit-learn
- genre stereotype in word embedding: https://arxiv.org/pdf/1606.06121v1.pdf
- twitter intent: https://twitter.com/intent/user?user_id=328567812
- google n-gram: https://books.google.com/ngrams/graph?content=she+is+a+nurse%2C+he+is+a+nurse&year_start=1800&year_end=2000&corpus=15&smoothing=3&share=&direct_url=t1%3B%2Cshe%20is%20a%20nurse%3B%2Cc0%3B.t1%3B%2Che%20is%20a%20nurse%3B%2Cc0
- FE techniques in simple words: https://codesachin.wordpress.com/2016/06/25/non-mathematical-feature-engineering-techniques-for-data-science/
- Money laundering detection: http://conf.startup.ml/blog/aml
- bias embedding: http://nlpers.blogspot.hr/2016/06/language-bias-and-black-sheep.html

24.6

- keynote: http://tpq.io/p/pyconsg.html#/
- customer segmentation: https://github.com/maoting1223/pycon_sg_2016
- https://github.com/mirri66/geodata

23.6

- I'm a speaker at pyconsg 2016: https://pycon.sg/schedule/
- https://github.com/airbnb/caravel
- googlenet: 22 layers inception http://arxiv.org/abs/1409.4842

22.6

- http://varianceexplained.org/r/year_data_scientist/
- conflicted ds: https://www.youtube.com/watch?v=7h2S3eM1OYQ&feature=youtu.be
- ICML 2016: http://icml.cc/2016/?page_id=1839
- My russian friends: https://alexanderdyakonov.wordpress.com/2016/05/31/avito-telstra-bnp/
- https://www.youtube.com/watch?v=1HrkBzLBJQg

21.6

- book of Andrew Ng: http://www.mlyearning.org/
- why we need so many classifiers: http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf
- variety of models to choose: https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms

20.6

- deephack, https://drive.google.com/file/d/0B0PX5JnpNX8yQ184Z3kwWmYyQUU/view?pref=2&pli=1
- mikilov rnn + scrnn: https://drive.google.com/file/d/0B0PX5JnpNX8yQ184Z3kwWmYyQUU/view?pref=2&pli=1


18.6

- deep hack: http://deepqa.tilda.ws/page78823.html
- Quoc Le, DL for lang understanding: https://www.youtube.com/watch?v=KmOdBS4BXZ0
- https://drive.google.com/file/d/0B0PX5JnpNX8yQ184Z3kwWmYyQUU/view?pref=2&pli=1
- https://drive.google.com/file/d/0BwJbEyAV32gETHR4YmdjcW5JUlU/view?pref=2&pli=1

17.6

- http://www.cs.utexas.edu/~roller/naacl2016dl.txt

16.6

toread:

- relationship modeling network https://github.com/miyyer/rmn
- QA compose NN: http://arxiv.org/pdf/1601.01705v4.pdf

- gensim 0.13 changelog: https://github.com/RaRe-Technologies/gensim/blob/develop/CHANGELOG.txt
- https://building-babylon.net/2015/06/03/document-embedding-with-paragraph-vectors/
- gbt http://nbviewer.jupyter.org/github/ogrisel/notebooks/blob/master/sklearn_demos/Gradient%20Boosting.ipynb
- https://docs.google.com/presentation/d/1KOvkwjZM1Wjj7hfbBP11fLrSUYJTPl_zaAM-fZje328/edit?pref=2&pli=1#slide=id.gd894eccf5_0_75
- https://foxtype.com/sentence-tree


15.6

- map 140M tweet: http://www.mapd.com/demos/tweetmap/
- http://blog.yhat.com/posts/rodeo-2.0-release.html

13.6

- personality: https://personality-insights-livedemo.mybluemix.net/
- imbalanced data https://github.com/ngaude/kaggle/blob/master/cdiscount/ImbalancedLearning.pdf
- just remember: https://ipgp.github.io/scientific_python_cheat_sheet/
- work on postgres https://github.com/dbcli/pgcli
- productionize with Kafka: http://blog.parsely.com/post/3886/pykafka-now/

11.6

- xgboost + nn: https://www.import.io/post/how-to-win-a-kaggle-competition/


9.6

user classifiers:

- http://www.slideshare.net/TedXiao/winning-kaggle-101-dmitry-larkos-experiences
- Humanizr: http://networkdynamics.org/resources/software/humanizr/
- tweet coder: http://networkdynamics.org/resources/software/tweetcoder/
- latent user, delip rao: http://www.cs.jhu.edu/~delip/smuc.pdf

Readings:

- intro prob in ipython: http://nbviewer.jupyter.org/url/norvig.com/ipython/Probability.ipynb
- thesis learning algos from data: http://www.cs.nyu.edu/media/publications/zaremba_wojciech.pdf
- Practical tools for exploring data and models: http://had.co.nz/thesis/practical-tools-hadley-wickham.pdf

8.6

- deep learning demo: http://www.somatic.io/models/V7Zx4Z9A
- ml debug: https://www.quora.com/Whats-the-best-way-to-debug-natural-language-processing-code-How-do-we-know-its-running-as-we-assume-I-ask-this-question-because-I-read-one-post-titled-as-what-is-the-best-way-to-test-machine-learning-code-I-am-working-on-one-natural-language-processing-task-and-has-confusion-on-how-to-debug-NLP
- confusion matrix: http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide-part-4/
- nlp ml error analysis tool: http://www.aclweb.org/anthology/C14-2001
- deepnet online: https://github.com/anujgupta82/DeepNets/blob/master/Online_Learning/Incorporating_feedback_in_DeepNets.ipynb
- sgd + elasticnet penalty better? https://www.quora.com/Are-there-any-real-applications-of-using-Elastic-Net
- http://cs231n.github.io/linear-classify/
- binary classification dog vs cat: http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- deeptext: am I scared? https://www.linkedin.com/pulse/what-listening-look-facebooks-ai-engine-radim-%C5%99eh%C5%AF%C5%99ek
- why Google open source Tensorflow: https://www.youtube.com/watch?v=Rnm83GqgqPE

7.6

- top 10 NLP con: http://idibon.com/the-top-10-nlp-conferences/
- metricle: https://metricle.com/api
- dark web svm, word2vec: https://homepages.staff.os3.nl/~delaat/rp/2014-2015/p99/report.pdf
- M$ deep learning: http://research.microsoft.com/pubs/246721/NAACL-HLT-2015_tutorial.pdf
- MS sgd trick: http://research.microsoft.com/pubs/192769/tricks-2012.pdf
- 10K class and 10M samples, OVR and SGD is best: https://hal.inria.fr/hal-00835810/PDF/TPAMI_minor_revision.pdf

6.6

- xin rong presentation: https://www.youtube.com/watch?v=D-ekE-Wlcds
- wevi: https://docs.google.com/presentation/d/1yQWN1CDWLzxGeIAvnGgDsIJr5xmy4dB0VmHFKkLiibo/
- stats for hacker pycon2016: https://speakerdeck.com/pycon2016/jake-vanderplas-statistics-for-hackers
- https://medium.com/udacity/this-week-in-machine-learning-3-june-2016-7f089ce984e7#.zbv7h9nyo
- word galaxy: http://www.anthonygarvan.com/wordgalaxy/
- pycon2016: https://github.com/singingwolfboy/build-a-flask-api
- http://burhan.io/flask-web-api-with-firebase/
- http://web.stanford.edu/class/cs224u/materials/cs224u-vsm-overview.pdf


1.6


- lab41: http://www.lab41.org/a-tour-of-sentiment-analysis-techniques-getting-a-baseline-for-sunny-side-up/
- doc2vec at tripadvisor: https://github.com/hellozeyu/An-advisor-for-TripAdvisor
- https://nycdatascience.com/an-advisor-for-tripadvisor/

10 lesson learned from Xavier recap:

- implicit signal beats explicit ones (almost always): clickbait, rating psychology
- your model will learn what you teach it to learn: feature, function, f score
- sup + unsup = life
- everything is ensemble
- model sequences: output of the model is input of others
- FE: reusable, transformable, interpretable, reliable
- ML infra: experimentation phase: easiness, flexibility, reusability. production phase: performance, scalable
- Debugging feature values
- you don't need to distribute ML algo
- DS + ML engineering = perfection


31.5

- pycon2016: https://www.youtube.com/channel/UCwTD5zJbsQGJN75MwbykYNw
- andreas, intro ML/sklearn for DS: https://github.com/amueller/introduction_to_ml_with_python
- Berkeley ds intro: https://data-8.appspot.com/sp16/course

30.5

- dirichlet process: http://stiglerdiet.com/blog/2015/Jul/28/dirichlet-distribution-and-dirichlet-process/
- pycon 2016: https://github.com/justmarkham/pycon-2016-tutorial/
- romance in word2vec: http://www.ghostweather.com/files/word2vecpride/
- topic quality coherence: http://palmetto.aksw.org/palmetto-webapp/
- https://spacy.io/docs
- https://spacy.io/docs/tutorials/twitter-filter
- http://sebastianraschka.com/Articles/2014_naive_bayes_1.html
- https://github.com/justmarkham/pycon-2016-tutorial


29.5

- cry analysis: http://www.robinwe.is/explorations/cry.html
- spacy preprocessing: https://github.com/cemoody/lda2vec/blob/master/lda2vec/preprocess.py
- spacy Tweet: https://spacy.io/docs/tutorials/twitter-filter
- lda2vec: full http://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=
- probalistic approach: http://chirayukong.github.io/infsci2725/resources/09_Probabilistic_Approaches.pdf
- lda curation: https://datawarrior.wordpress.com/2016/04/20/local-and-global-words-and-topics/
- why hdbscan: http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb
- auto ml: http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
- http://www.kdnuggets.com/2016/05/five-machine-learning-projects-cant-overlook.html
- topic2vec: https://www.cs.cmu.edu/~diyiy/docs/naacl15.pdf


26.5

- http://alexperrier.github.io/jekyll/update/2015/09/04/topic-modeling-of-twitter-followers.html
- http://alexperrier.github.io/jekyll/update/2015/09/16/segmentation_twitter_timelines_lda_vs_lsa.html
- https://begriffs.com/posts/2015-03-10-better-tweets-datascience.html
- https://github.com/alexperrier/datatalks/tree/master/twitter
- https://issuu.com/andriusknispelis/docs/topic_models_-_video
- http://www.aclweb.org/anthology/W15-1526
- https://www.opendatascience.com/blog/dissecting-the-presidential-debates-with-an-nlp-scalpel/
- https://speakerdeck.com/bmabey/visualizing-topic-models

25.5

In summary, here is what I recommend if you plan to use word2vec: choose the right training parameters and training data for word2vec, use avg predictor for query, sentence and paragraph(code here) after picking a dominant word set and apply deep learning on resulted vectors.

===

For SGNS, here is what I believe really happens during the training:
If two words appear together, the training will try to increase their cosine similarity. If two words never appear together, the training will reduce their cosine similarity. So if there are a lot of user queries such as “auto insurance” and “car insurance”, then “auto” vector will be similar to “insurance” vector (cosine similarity ~= 0.3) and “car” vector will also be similar to “insurance” vector. Since “insurance”, “loan” and “repair” rarely appear together in the same context, their vectors have small mutual cosine similarity (cosine similarity ~= 0.1). We can treat them as orthogonal to each other and think them as different dimensions. After training is complete, “auto” vector will be very similar to “car” vector (cosine similarity ~= 0.6) because both of them are similar in “insurance” dimension, “loan” dimension and “repair” dimension.   This intuition will be useful if you want to better design your training data to meet the goal of your text learning task.

===

for short sentences/phrases, Tomas Mikolov recommends simply adding up individual vector words to get a "sentence vector" (see his recent NIPS slides). 

For longer documents, it is an open research question how to derive their representation, so no wonder you're having trouble :)

I like the way word2vec is running (no need to use important hardware to process huge collection of text). It's more usable than LSA or any system which requires a term-document matrix.

Actually LSA requires less structured data (only a bag-of-words matrix, whereas word2vec requires exact word sequences), so there's no fundamental difference in input complexity.


- http://douglasduhaime.com/blog/clustering-semantic-vectors-with-python 
- https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
- http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html
- http://www.aclweb.org/anthology/W15-1526
- ML model: http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/
 


24.5

- https://github.com/edburnett/twitter-text-python
- http://eng.kifi.com/from-word2vec-to-doc2vec-an-approach-driven-by-chinese-restaurant-process/
- https://www.insight-centre.org/sites/default/files/publications/14.033_insight-snow14dc-final.pdf
- http://rali.iro.umontreal.ca/rali/sites/default/files/publis/Atefeh_et_al-2013-Computational_Intelligence-2.pdf

TSNE:

- javascript: http://karpathy.github.io/2014/07/02/visualizing-top-tweeps-with-t-sne-in-Javascript/
- http://cs.stanford.edu/people/karpathy/tsnejs/index.html

Conferences:

- word2vec tree: https://github.com/pvthuy/word2vec-visualization
- flask, api, mongo, d3: http://adilmoujahid.com/posts/2015/01/interactive-data-visualization-d3-dc-python-mongodb/
- https://github.com/RaRe-Technologies/movie-plots-by-genre
- wmd: http://vene.ro/blog/word-movers-distance-in-python.html
- word2vec viz: https://ronxin.github.io/wevi/
- news analytics in finance: https://vimeo.com/67901816
- table2vec: http://www.slideshare.net/SparkSummit/using-data-science-to-transform-opentable-into-delgado-das
- data by the bay: http://data.bythebay.io/schedule.html
- pydataberlin: http://pydata.org/berlin2016/

20.5

- scatter with images: https://gist.github.com/lukemetz/be6123c7ee3b366e333a

19.5

- wise 203 classes, vocab = 300k, sample = 64k, test = 34j=k, http://alexanderdyakonov.narod.ru/wise2014-kaggle-Dyakonov.pdf
- yelp review to multi label: food, deal, ambience,... http://www.ics.uci.edu/~vpsaini/
- instagram: http://instagram-engineering.tumblr.com/post/117889701472/emojineering-part-1-machine-learning-for-emoji
- emoji embedding http://www.danielforsyth.me/nba-twitter-emojis-and-word-embeddings/
- tweetmap in websummit event: http://blog.aylien.com/post/133931414053/analyzing-tweets-from-web-summit-2015-semantic
- topic2vec: http://arxiv.org/pdf/1506.08422.pdf
- http://googleresearch.blogspot.com/2016/05/chat-smarter-with-allo.html
- https://en.wikipedia.org/wiki/Limited-memory_BFGS

18.5

- building data processing at budget: http://www.slideshare.net/GaelVaroquaux/building-a-cuttingedge-data-processing-environment-on-a-budget
- https://radimrehurek.com/gensim/wiki.html
- calibration: http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#example-calibration-plot-calibration-curve-py

- http://glowingpython.blogspot.com/2014/02/terms-selection-with-chi-square.html
- which feature selection: http://sebastianraschka.com/faq/docs/feature_sele_categories.html
- which learning algos: http://sebastianraschka.com/faq/docs/best-ml-algo.html
- for intepretability use tree: http://sebastianraschka.com/faq/docs/model-selection-in-datascience.html
- LR vs NB: http://sebastianraschka.com/faq/docs/naive-bayes-vs-logistic-regression.html
- yelp review classifier: https://github.com/parulsingh/FlaskAppCS194
- ngsg is not mf yet: https://building-babylon.net/2016/05/12/skipgram-isnt-matrix-factorisation/
- http://blog.aylien.com/post/133931414053/analyzing-tweets-from-web-summit-2015-semantic
- http://aylien.com/web-summit-2015-tweets-part1

sentifi:

- https://github.com/bdhingra/tweet2vec
- tweet2vec https://arxiv.org/abs/1605.03481
- syntaxnet: https://github.com/tensorflow/models/tree/master/syntaxnet
- hijack compromise user account http://www.icir.org/vern/papers/twitter-compromise.ccs2014.pdf
- user classification: name + loc http://www.cs.jhu.edu/~vandurme/papers/broadly-improving-user-classfication-via-communication-based-name-and-location-clustering-on-twitter.pdf
- chrispot: http://sentiment.christopherpotts.net/tokenizing.html
- https://github.com/cbuntain/TwitterFergusonTeachIn
- mining tweet: https://rawgit.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/master/ipynb/html/Chapter%201%20-%20Mining%20Twitter.html
- NE: https://noisy-text.github.io/pdf/WNUT10.pdf
- tokenizer: http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py
- twitter tokenizer online: http://sentiment.christopherpotts.net/tokenizing/results/
- cs224u understanding nlp: http://nbviewer.jupyter.org/github/cgpotts/cs224u/
- https://spacy.io/blog/german-model?utm_source=News&utm_campaign=87a64aae50-German_release_newsletter&utm_medium=email&utm_term=0_89ad33e698-87a64aae50-64293797
- jupyter theme: http://sherifsoliman.com/2016/01/11/theming-ipython-jupyter-notebook/
- noisy text need to be normalized: https://noisy-text.github.io/norm-shared-task.html
- understanding user profile/twitter: https://blog.twitter.com/2015/guest-post-understanding-users-through-twitter-data-and-machine-learning
- word2vec with numba: https://d10genes.github.io/blog/2016/05/03/word2vec/
- analyzing text data at Firefox: http://web.stanford.edu/~rjweiss/public_html/MozFest2013/
- pretrained word2vec https://github.com/3Top/word2vec-api
- twitter music word2vec: http://www.netbase.com/blog/understanding-beliebers-word2vec-twitter/
- text + images with CNN: https://www.scribd.com/doc/305710656/Convolutional-Neural-Networks-for-Multimedia-Sentiment-Analysis
- feature pivot: http://www.hpl.hp.com/techreports/2011/HPL-2011-98.pdf
- nlp with cnn: http://www.slideshare.net/devashishshanker/deep-learning-for-natural-language-processing
- event detection http://www.hpl.hp.com/techreports/2011/HPL-2011-98.pdf
- http://www.zdnet.com/article/big-data-what-to-trust-data-science-or-the-bosss-sixth-sense/
- tf is winning: https://medium.com/@mjhirn/tensorflow-wins-89b78b29aafb#.6lebzwbyx
- a vc blog: http://avc.com
- hijacking: http://www.icir.org/vern/papers/twitter-compromise.ccs2014.pdf
- us president prediction: http://www.aioptify.com/predictinguselection.php
- https://thestack.com/world/2015/05/08/three-steps-to-building-a-twitter-driven-trading-bot/
- http://file.scirp.org/pdf/SN_2015070917142293.pdf
- tweet latent attributes: http://boingboing.net/2014/09/01/twitter-uses-an-algorithm-to-f.html
- user gender inference: http://www.aclweb.org/anthology/W14-5408
- https://blog.bufferapp.com/the-5-types-of-tweets-to-keep-your-buffer-full-and-your-followers-engaged
- classifying user latent attributes: http://www.cs.jhu.edu/~delip/smuc.pdf
- http://myownhat.blogspot.com/
- http://bugra.github.io/work/notes/2015-01-17/mining-a-vc/
- NER with w2v, 400M tweet: http://www.fredericgodin.com/software/


http://davidrosenberg.github.io/ml2016/#home

pydatalondon 2016:

- http://www.thetalkingmachines.com
- https://www.youtube.com/user/PyDataTV
- pymc: https://docs.google.com/presentation/d/1QNxSjDHJbFL7vFwQHHheeGmBHEJAo39j28xdObFY6Eo/edit#slide=id.gdfcfebc22_0_118
- https://github.com/springcoil/PyDataLondonTutorial/blob/master/deck-17.pdf
- https://speakerdeck.com/bargava/introduction-to-deep-learning
- https://github.com/rouseguy/intro2deeplearning/
- https://github.com/rouseguy/intro2stats
- https://github.com/kylemcdonald/SmileCNN
- https://github.com/springcoil/PyDataLondonTutorial/blob/master/notebooks/Statistics.ipynb
- http://greenteapress.com/complexity/thinkcomplexity.pdf
- http://matthewearl.github.io/2016/05/06/cnn-anpr/

spotify:

- http://www.slideshare.net/AndySloane/machine-learning-spotify-madison-big-data-meetup
- http://www.slideshare.net/erikbern/music-recommendations-mlconf-2014

lda asyn, auto alpha: http://rare-technologies.com/python-lda-in-gensim-christmas-edition/

mapk: https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics

ilcr2016: https://tensortalk.com/?cat=conference-iclr-2016

l.m.thang

- https://github.com/lmthang/nlm
- http://nlp.stanford.edu/~lmthang/data/papers/iclr16_multi.pdf


https://github.com/jxieeducation/DIY-Data-Science

http://drivendata.github.io/cookiecutter-data-science/

http://ofey.me/papers/sparse_ijcai16.pdf

Spotify:

- https://github.com/mattdennewitz/playlist-to-vec
- http://wonder.fm/
- https://social.shorthand.com/huntedguy/3CfQA8mj2S/playlist-harvesting

skflow:

- https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.7a7s8tkke
- https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.jgxmezy95
- https://medium.com/@ilblackdragon/tensorflow-tutorial-part-3-c5fc0662bc08#.2d22an1xp
- http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/
- https://libraries.io/github/mhlr/skflow
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples

a few useful things to know about ML:

- https://blog.bigml.com/2013/02/15/everything-you-wanted-to-know-about-machine-learning-but-were-too-afraid-to-ask-part-one/
- https://blog.bigml.com/2013/02/21/everything-you-wanted-to-know-about-machine-learning-but-were-too-afraid-to-ask-part-two/

tdb: https://github.com/ericjang/tdb

dask for task parallel, delayed: http://dask.pydata.org/en/latest/examples-tutorials.html

skflow: 

- pip install git+git://github.com/tensorflow/skflow.git
- http://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html


http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/

https://medium.com/a-year-of-artificial-intelligence/lenny-2-autoencoders-and-word-embeddings-oh-my-576403b0113a#.ecj0iv4n8

https://github.com/andrewt3000/DL4NLP/blob/master/README.md

tf:

- http://terryum.io/ml_applications/2016/04/25/TF-Code-Structure/
- http://www.slideshare.net/tw_dsconf/tensorflow-tutorial

tf chatbot: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot

- deep inversion : https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb
- encoder decoder with attention: http://arxiv.org/pdf/1512.01712v1.pdf
- keras tut: http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/tutorials/keras.pdf

Bayesian Opt: https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb

click-o-tron rnn: http://clickotron.com
auto generated headline clickbait: https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/

http://blog.computationalcomplexity.org/2016/04/the-master-algorithm.html
http://jyotiska.github.io/blog/posts/python_libraries.html

LSTM: http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

CS224d: 

- TF intro: http://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
- RNN: http://cs224d.stanford.edu/lectures/CS224d-Lecture8.pdf

Sota of sa, mikolo and me :) 

- http://arxiv.org/pdf/1412.5335.pdf
- https://github.com/mesnilgr/iclr15/tree/master/scripts

Thang M. L: http://web.stanford.edu/class/cs224n/handouts/cs224n-lecture16-nmt.pdf

CS224d reports:

- classify online forum answer/non-answer: https://cs224d.stanford.edu/reports/AbajianAaron.pdf
- gender classification: https://cs224d.stanford.edu/reports/BartleAric.pdf
- job prediction: https://cs224d.stanford.edu/reports/BoucherEric.pdf
- text sum: https://cs224d.stanford.edu/reports/ChaiElaina.pdf
- email spam: https://cs224d.stanford.edu/reports/EugeneLouis.pdf
- jp2en: https://cs224d.stanford.edu/reports/GreensteinEric.pdf
- improve PV: https://cs224d.stanford.edu/reports/HongSeokho.pdf
- twitter sa: https://cs224d.stanford.edu/reports/YuanYe.pdf
- yelp sa: https://cs224d.stanford.edu/reports/YuApril.pdf
- author detector: https://cs224d.stanford.edu/reports/YaoLeon.pdf
- IMDB to Yelp: https://cs224d.stanford.edu/reports/XingMargaret.pdf
- Reddit: https://cs224d.stanford.edu/reports/TingJason.pdf
- Quora: https://cs224d.stanford.edu/reports/JindalPranav.pdf

QA in keras:

- https://github.com/avisingh599/visual-qa/blob/master/scripts/trainMLP.py
- https://avisingh599.github.io/deeplearning/visual-qa/

Chinese LSTM + word2vec: 

- https://github.com/taozhijiang/chinese_nlp/blob/master/DL_python/dl_segment_v2.py
- https://github.com/taozhijiang/chinese_nlp


DL with SA: https://cs224d.stanford.edu/reports/HongJames.pdf

MAB:

- mab book: http://pdf.th7.cn/down/files/1312/bandit_algorithms_for_website_optimization.pdf
- yhat: http://blog.yhat.com/posts/the-beer-bandit.html
- test significance with AB, conversation rate opt with MAB: https://vwo.com/blog/multi-armed-bandit-algorithm/
- when to use multiarmed bandits: http://conversionxl.com/bandit-tests/
- multibandit: http://stevehanov.ca/blog/index.php?id=132

cnn nudity detection: http://blog.clarifai.com/what-convolutional-neural-networks-see-at-when-they-see-nudity/#.VxbdB0xcSko

sigopt: https://github.com/sigopt/sigopt_sklearn

first contact with TF: http://www.jorditorres.org/first-contact-with-tensorflow/

eval of ML using A/B or multibandit: http://blog.dato.com/how-to-evaluate-machine-learning-models-the-pitfalls-of-ab-testing

how to make mistakes in Python: www.oreilly.com/programming/free/files/how-to-make-mistakes-in-python.pdf

keras tut: https://uwaterloo.ca/data-science/sites/ca.data-science/files/uploads/files/keras_tutorial.pdf

Ogrisel word embedding: https://speakerd.s3.amazonaws.com/presentations/31f18ad0522c0132b9b662e7bb117668/Word_Embeddings.pdf

Tensorflow whitepaper: http://download.tensorflow.org/paper/whitepaper2015.pdf

Arimo distributed tensorflow: https://arimo.com/machine-learning/deep-learning/2016/arimo-distributed-tensorflow-on-spark/

Best ever word2vec in code: http://nbviewer.jupyter.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb

TF japanese: http://www.slideshare.net/yutakashino/tensorflow-white-paper

TF tut101: https://github.com/aymericdamien/TensorFlow-Examples

Jeff Dean: http://learningsys.org/slides/NIPS-Learning-Systems-Workshop-TensorFlow-Jeff-Dean.pdf
DL: http://www.thoughtly.co/blog/deep-learning-lesson-1/
Distributed TF: https://www.tensorflow.org/versions/r0.8/how_tos/distributed/index.html

playground: http://playground.tensorflow.org/

Hoang Duong blog: http://hduongtrong.github.io/
Word2vec short explanation: http://hduongtrong.github.io/2015/11/20/word2vec/

ForestSpy: https://github.com/jvns/forestspy/blob/master/inspecting%20random%20forest%20models.ipynb

- keras for mnist: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
- lasagne installation https://martin-thoma.com/lasagne-for-python-newbies/

Netflix:

- http://www.wired.com/2012/04/netflix-prize-costs/
- http://www.wired.com/2009/09/bellkors-pragmatic-chaos-wins-1-million-netflix-prize/


Lessons learned

- http://machinelearningmastery.com/lessons-learned-building-machine-learning-systems/
- http://techjaw.com/2015/02/11/10-machine-learning-lessons-harnessed-by-netflix/
- https://medium.com/@xamat/10-more-lessons-learned-from-building-real-life-ml-systems-part-i-b309cafc7b5e#.klowhfq10

WMD:

- word mover distance: https://github.com/mkusner/wmd
- gensim wmd: https://speakerdeck.com/tmylk/same-content-different-words

Hanoi trip:

- tensorflow scan: learn the cum sum https://nbviewer.jupyter.org/github/rdipietro/tensorflow-notebooks/blob/master/tensorflow_scan_examples/tensorflow_scan_examples.ipynb
- https://jayantj.github.io/posts/project-gutenberg-word2vec
- stacking: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
- Learn and think like human: http://arxiv.org/pdf/1604.00289v1.pdf
- predictive modeling + AI: https://speakerd.s3.amazonaws.com/presentations/30ad41b99258471f9485118f904f8cfb/predictive_modeling_and_deep_learning.pdf
- sklearn vs tf: https://github.com/rasbt/python-machine-learning-book/blob/master/faq/tensorflow-vs-scikitlearn.md
- advances in DL for NLP: http://cs.nyu.edu/~zaremba/docs/Advances%20in%20deep%20learning%20for%20NLP.pdf
- Xavier, 10 lessons learned: https://medium.com/@xamat/10-more-lessons-learned-from-building-real-life-ml-systems-part-i-b309cafc7b5e#.klowhfq10

- pizza analysis: http://yoavz.com/potd/
- R at airbnb: https://medium.com/airbnb-engineering/using-r-packages-and-education-to-scale-data-science-at-airbnb-906faa58e12d#.deo3t37vr
- 450 hours in data science: http://studiy.co/path/data-science/
- LR + SGD + FM: https://gist.github.com/kalaidin/9ea737ad771fcf073e57
- libFM: http://www.ics.uci.edu/~smyth/courses/cs277/papers/factorization_machines_with_libFM.pdf
- intro FM: http://www.slideshare.net/0x001/intro-to-factorization-machines
- fastFM: https://github.com/ibayer/fastFM
- winning data science competition: https://speakerdeck.com/datasciencela/jeong-yoon-lee-winning-data-science-competitions-data-science-meetup-oct-2015
- python for data analyst: https://www.kevinsheppard.com/images/0/09/Python_introduction.pdf
- risk modeling: https://risk-engineering.org/static/PDF/slides-stat-modelling.pdf
- pyfm: https://github.com/coreylynch/pyFM
- mlss2014: http://www.mlss2014.com/materials.html
- xavier: https://www.slideshare.net/slideshow/embed_code/key/gt6HuUzZ4Z7flf
- Pedro: http://www.thetalkingmachines.com/blog/
- Machine Intelligence 2.0: https://cdn-images-1.medium.com/max/2000/1*A9exqeQ69XjjSJgMyDEo6Q.jpeg
- Quora - all about data scientits: https://www.quora.com/What-are-the-best-blogs-for-data-scientists-to-read
- World of though vector: http://www.pamitc.org/cvpr15/files/lecun-20150610-cvpr-keynote.pdf
- newbie nlp lab: https://github.com/piskvorky/topic_modeling_tutorial/
- why and when log-log is used: http://www.forbes.com/sites/naomirobbins/2012/01/19/when-should-i-use-logarithmic-scales-in-my-charts-and-graphs/#41c6dc0c3cd8
- lzma: https://parezcoydigo.wordpress.com/2011/10/09/clustering-with-compression-for-the-historian/
- Tom Vincent: http://insightdatascience.com/blog/tom_vincent_qanda.html
- Normalized Compression Distance: http://tamediadigital.ch/2016/03/20/normalized-compression-distance-a-simple-and-useful-method-for-text-clustering-2/
- Yoav Goldberg: https://www.youtube.com/watch?v=xw5HL5h1wxY
- Sklearn production on Dato: https://www.youtube.com/watch?v=AwjeRg1u5VI

VinhKhuc:

- how many k for CV: k = N e.g. LOOCV http://vinhkhuc.github.io/2015/03/01/how-many-folds-for-cross-validation.html
- backprop http://vinhkhuc.github.io/2015/03/29/backpropagation.html
- qa bAbI task: https://github.com/vinhkhuc/MemN2N-babi-python
- lstm/rnn: http://vinhkhuc.github.io/2015/11/19/rnn-lstm.html

RS: 

- https://code.facebook.com/posts/861999383875667/recommending-items-to-more-than-a-billion-people/
- http://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/

Data science bootcamp: https://cambridgecoding.com/datascience-bootcamp#outline

CambridgeCoding NLP: 

- https://drive.google.com/file/d/0B_ZOKLUe_XPaNVFHM3M4dHRzV28/view?pli=1
- http://blog.cambridgecoding.com/2016/03/24/misleading-modelling-overfitting-cross-validation-and-the-bias-variance-trade-off/

Annoy:

- Annoy, Luigi: http://erikbern.com/, https://www.hakkalabs.co/articles/approximate-nearest-neighbors-vector-models
- LSH: https://speakerdeck.com/maciejkula/locality-sensitive-hashing-at-lyst
- http://www.slideshare.net/erikbern/approximate-nearest-neighbor-methods-and-vector-models-nyc-ml-meetup
- https://github.com/spotify/annoy

RPForest: https://github.com/lyst/rpforest
LightFM: https://github.com/lyst/lightfm
Secure because of math: https://www.youtube.com/watch?v=TYVCVzEJhhQ
Talking machines: http://www.thetalkingmachines.com/
Dive into DS: https://github.com/rasbt/dive-into-machine-learning

DS process: https://www.oreilly.com/ideas/building-a-high-throughput-data-science-machine
Friendship paradox: https://vuhavan.wordpress.com/2016/03/25/ban-ban-ban-nhieu-hon-ban-ban/

AB test:

- notebook: https://github.com/Volodymyrk/stats-testing-in-python/blob/master/01%20-%20Single%20Sample%20tests%20for%20Mean.ipynb
- https://medium.com/@rchang/my-two-year-journey-as-a-data-scientist-at-twitter-f0c13298aee6#.t1h9ouwpg
- http://multithreaded.stitchfix.com/blog/2015/05/26/significant-sample/
- http://nerds.airbnb.com/experiments-at-airbnb/
- https://www.quora.com/When-should-A-B-testing-not-be-trusted-to-make-decisions/answer/Edwin-Chen-1?srid=sL8&share=1

EMNLP 2015:

- semantic sim of embedding: https://www.cs.cmu.edu/~ark/EMNLP-2015/tutorials/34/34_OptionalAttachment.pdf
- social text analysis: https://www.cs.cmu.edu/~ark/EMNLP-2015/tutorials/3/3_OptionalAttachment.pdf
- personality research in NLP: https://www.cs.cmu.edu/~ark/EMNLP-2015/tutorials/2/2_OptionalAttachment.pdf

To read:

- https://github.com/rasbt/algorithms_in_ipython_notebooks
- https://www.blackhat.com/docs/webcast/02192015-secure-because-math.pdf
- http://nirvacana.com/thoughts/becoming-a-data-scientist/
- http://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/IndeedJobs.ipynb
- http://www.john-foreman.com/data-smart-book.html
- http://www.thetalkingmachines.com/blog/2015/4/23/starting-simple-and-machine-learning-in-meds
- https://github.com/justmarkham/DAT8
- https://github.com/donnemartin/data-science-ipython-notebooks

Idols:

- Alex Pinto: MLSec
- Peadar Coyle: https://peadarcoyle.wordpress.com/, https://github.com/springcoil/pydataamsterdamkeynote, http://slides.com/springcoil/dataproducts-11#/27, https://medium.com/@peadarcoyle/three-things-i-wish-i-knew-earlier-about-machine-learning-54cb0d23ca29#.uc6e049rl
- Radmim: gensim
- Delip Rao: http://deliprao.com/archives/129
- Alex: http://alexanderdyakonov.narod.ru/engcontests.htm
- Yorav: https://www.cs.bgu.ac.il/~yoavg/uni/
- Andreij: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Sebastian: http://www.kdnuggets.com/2016/02/conversation-data-scientist-sebastian-raschka-podcast.html
- Joel Grus: http://joelgrus.com/
- Bugra: http://bugra.github.io/

IPython/Jupyter:

- https://docs.google.com/presentation/d/1PHnnkKYgjq1lcSDaVyhZP0Fs7qC70iA07b2Jv0uisUE/mobilepresent?slide=id.g10d199ad72_0_20

LSTM:

- RNN for music: http://erikbern.com/2014/06/28/recurrent-neural-networks-for-collaborative-filtering/
- skflow: https://github.com/tensorflow/skflow/tree/master/examples
- dropout: http://arxiv.org/abs/1409.2329
- seq2seq: http://arxiv.org/abs/1409.3215
- simple char rnn: https://gist.github.com/karpathy/d4dee566867f8291f086
- https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html#the-model
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/

RNN:

- https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
- Char RNN: http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- http://karpathy.github.io/neuralnets/

Unicode:

- ascii fix: http://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
- http://nedbatchelder.com/text/unipain/unipain.html#45

EVENTS:

- April 8-10 2016: PyData Madrid 
- April 15-17 2016: PyData Florence 
- May 6-8 2016: PyData London hosted by Bloomberg 
- May 20-21 2016: PyData Berlin 
- September 14-16 2016: PyData Carolinas hosted by IBM 
- October 7-9 2016: PyData DC hosted by Capital One 
- November 28-30 2016: PyData Cologne 

Other Conference Dates Coming Soon!

- PyData Chicago
- PyData NYC
- PyData Paris
- PyData Silicon Valley
- pydata amsterdam: http://pydata.org/amsterdam2016/schedule/ https://speakerdeck.com/maciejkula/hybrid-recommender-systems-at-pydata-amsterdam-2016
- gcp 23-24 March
- pycon sg: June 23-25 
- emnlp: june, austin, us
- pydata

QUOTES:

- My name is Sherlock Homes. It is my business to know what other people dont know.
- Take the first step in faith. You don't have to see the whole staircase, just take the first step. [M.L.King. Jr]
- "Data data data" he cried impatiently. I can't make bricks without clay. [Arthur Donan Doyle]

STATS:

- http://vietsciences.free.fr/vietnam/bienkhao-binhluan/tuoithovuachuavn.htm


BOOKS:

- http://shop.oreilly.com/product/0636920033400.do
- https://web.stanford.edu/~hastie/StatLearnSparsity_files/SLS_corrected_1.4.16.pdf
- https://leanpub.com/interviewswithdatascientists

CLUSTER:

- distance: http://stackoverflow.com/questions/22433884/python-gensim-how-to-calculate-document-similarity-using-the-lda-model#answer-22756647
- hac with lsi: https://groups.google.com/forum/#!topic/gensim/0Ev8Okf3MCs
- clustering eva: http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- Silhouette analysis: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- https://groups.google.com/forum/#!msg/gensim/ZxauGgh9Vqs/prIMalR8LbgJ
- http://www.site.uottawa.ca/~diana/csi5180/TextClustering.pdf
- http://stackoverflow.com/questions/17537722/better-text-documents-clustering-than-tf-idf-and-cosine-similarity
- http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/
- 30 day indexed: http://googlenewsblog.blogspot.com/2008/05/keeping-good-news-stories-together-just.html
- http://www.mondaynote.com/2013/02/24/google-news-the-secret-sauce/
- http://searchengineland.com/google-news-ranking-stories-30424
- http://nsuworks.nova.edu/cgi/viewcontent.cgi?article=1051&context=gscis_etd
- https://github.com/lmcinnes/hdbscan
- http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.6.ipynb
- http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations%202D%20v0.6.ipynb
- http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb

EMBEDDING:

- https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/
- http://www.offconvex.org/2016/02/14/word-embeddings-2/
- improving sem embedding words rep: https://levyomer.wordpress.com/2015/03/30/improving-distributional-similarity-with-lessons-learned-from-word-embeddings/
- whiskey: http://wrec.herokuapp.com/methodology
- lda: topic eva: http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
- lda2vec: http://www.slideshare.net/ChristopherMoody3/word2vec-lda-and-introducing-a-new-hybrid-algorithm-lda2vec-57135994
- http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty_newsgroups/lda.ipynb
- text2vec: http://dsnotes.com/articles/glove-enwiki
- Swivel Submatrix Wise Vector Embedding Learner http://arxiv.org/pdf/1602.02215v1.pdf
- https://sense2vec.spacy.io/?natural_language_processing%7CNOUN

Linux:

- http://randyzwitch.com/gnu-parallel-medium-data/

BENCHMARK:

- keras vs theano vs tensorflow: https://www.reddit.com/r/MachineLearning/comments/462p41/pros_and_cons_of_keras_vs_lasagne_for_deep/
- http://felixlaumon.github.io/2015/01/08/kaggle-right-whale.html
- https://github.com/zer0n/deepframeworks/blob/master/README.md
- https://github.com/soumith/convnet-benchmarks/issues/66
- https://github.com/soumith/convnet-benchmarks/blob/master/README.md
- https://github.com/lmcinnes/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations.ipynb
- https://github.com/szilard/benchm-ml
- word2vec: http://rare-technologies.com/parallelizing-word2vec
