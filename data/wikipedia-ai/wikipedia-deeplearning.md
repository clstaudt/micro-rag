[alt=Representing images on multiple layers of abstraction in deep
learning|thumb|upright=1.35|Representing images on multiple layers of
abstraction in deep learning[1]](File:Deep_Learning.jpg "wikilink")

**Deep learning** is the subset of [machine
learning](machine_learning "wikilink") methods based on [neural
networks](neural_network_(machine_learning) "wikilink") with
[representation learning](representation_learning "wikilink"). The
adjective "deep" refers to the use of multiple layers in the network.
Methods used can be either [supervised](Supervised_learning "wikilink"),
[semi-supervised](Semi-supervised_learning "wikilink") or
[unsupervised](Unsupervised_learning "wikilink").[2]

Deep-learning architectures such as [deep neural
networks](#Deep_neural_networks "wikilink"), [deep belief
networks](deep_belief_network "wikilink"), [recurrent neural
networks](recurrent_neural_networks "wikilink"), [convolutional neural
networks](convolutional_neural_networks "wikilink") and
[transformers](Transformer_(machine_learning_model) "wikilink") have
been applied to fields including [computer
vision](computer_vision "wikilink"), [speech
recognition](speech_recognition "wikilink"), [natural language
processing](natural_language_processing "wikilink"), [machine
translation](machine_translation "wikilink"),
[bioinformatics](bioinformatics "wikilink"), [drug
design](drug_design "wikilink"), [medical image
analysis](medical_image_analysis "wikilink"), [climate
science](Climatology "wikilink"), material inspection and [board
game](board_game "wikilink") programs, where they have produced results
comparable to and in some cases surpassing human expert
performance.[3][4][5]

Early forms of neural networks were inspired by information processing
and distributed communication nodes in [biological
systems](biological_system "wikilink"), in particular the [human
brain](human_brain "wikilink"). However, current neural networks do not
intend to model the brain function of organisms, and are generally seen
as low quality models for that purpose.[6]

## Overview

Most modern deep learning models are based on multi-layered [neural
networks](neural_network_(machine_learning) "wikilink") such as
[convolutional neural networks](convolutional_neural_network "wikilink")
and [transformers](transformer_(neural_network) "wikilink"), although
they can also include [propositional
formulas](propositional_formula "wikilink") or latent variables
organized layer-wise in deep [generative
models](generative_model "wikilink") such as the nodes in [deep belief
networks](deep_belief_network "wikilink") and deep [Boltzmann
machines](Boltzmann_machine "wikilink").[7]

Fundamentally, deep learning refers to a class of [machine
learning](machine_learning "wikilink")
[algorithms](algorithm "wikilink") in which a hierarchy of layers is
used to transform input data into a slightly more abstract and composite
representation. For example, in an [image
recognition](image_recognition "wikilink") model, the raw input may be
an [image](image "wikilink") (represented as a
[tensor](Tensor_(machine_learning) "wikilink") of
[pixels](Pixel "wikilink")). The first representational layer may
attempt to identify basic shapes such as lines and circles, the second
layer may compose and encode arrangements of edges, the third layer may
encode a nose and eyes, and the fourth layer may recognize that the
image contains a face.

Importantly, a deep learning process can learn which features to
optimally place in which level *on its own*. Prior to deep learning,
machine learning techniques often involved hand-crafted [feature
engineering](feature_engineering "wikilink") to transform the data into
a more suitable representation for a classification algorithm to operate
upon. In the deep learning approach, features are not hand-crafted and
the model [discovers](representation_learning "wikilink") useful feature
representations from the data automatically. This does not eliminate the
need for hand-tuning; for example, varying numbers of layers and layer
sizes can provide different degrees of abstraction.[8][9]

The word "deep" in "deep learning" refers to the number of layers
through which the data is transformed. More precisely, deep learning
systems have a substantial *credit assignment path* (CAP) depth. The CAP
is the chain of transformations from input to output. CAPs describe
potentially causal connections between input and output. For a
[feedforward neural network](feedforward_neural_network "wikilink"), the
depth of the CAPs is that of the network and is the number of hidden
layers plus one (as the output layer is also parameterized). For
[recurrent neural networks](recurrent_neural_network "wikilink"), in
which a signal may propagate through a layer more than once, the CAP
depth is potentially unlimited.[10] No universally agreed-upon threshold
of depth divides shallow learning from deep learning, but most
researchers agree that deep learning involves CAP depth higher than 2.
CAP of depth 2 has been shown to be a universal approximator in the
sense that it can emulate any function.[11] Beyond that, more layers do
not add to the function approximator ability of the network. Deep models
(CAP &gt; 2) are able to extract better features than shallow models and
hence, extra layers help in learning the features effectively.

Deep learning architectures can be constructed with a
[greedy](greedy_algorithm "wikilink") layer-by-layer method.[12] Deep
learning helps to disentangle these abstractions and pick out which
features improve performance.[13]

Deep learning algorithms can be applied to unsupervised learning tasks.
This is an important benefit because unlabeled data are more abundant
than the labeled data. Examples of deep structures that can be trained
in an unsupervised manner are [deep belief
networks](deep_belief_network "wikilink").[14][15]

## Interpretations

Deep neural networks are generally interpreted in terms of the
[universal approximation
theorem](universal_approximation_theorem "wikilink")[16][17][18][19][20]
or [probabilistic
inference](Bayesian_inference "wikilink").[21][22][23][24][25]

The classic universal approximation theorem concerns the capacity of
[feedforward neural networks](feedforward_neural_networks "wikilink")
with a single hidden layer of finite size to approximate [continuous
functions](continuous_functions "wikilink").[26][27][28][29] In 1989,
the first proof was published by [George
Cybenko](George_Cybenko "wikilink") for
[sigmoid](sigmoid_function "wikilink") activation functions[30] and was
generalised to feed-forward multi-layer architectures in 1991 by Kurt
Hornik.[31] Recent work also showed that universal approximation also
holds for non-bounded activation functions such as [Kunihiko
Fukushima](Kunihiko_Fukushima "wikilink")'s [rectified linear
unit](rectified_linear_unit "wikilink").[32][33]

The universal approximation theorem for [deep neural
networks](deep_neural_network "wikilink") concerns the capacity of
networks with bounded width but the depth is allowed to grow. Lu et
al.[34] proved that if the width of a deep neural network with
[ReLU](ReLU "wikilink") activation is strictly larger than the input
dimension, then the network can approximate any [Lebesgue integrable
function](Lebesgue_integration "wikilink"); if the width is smaller or
equal to the input dimension, then a deep neural network is not a
universal approximator.

The [probabilistic](probabilistic "wikilink") interpretation[35] derives
from the field of [machine learning](machine_learning "wikilink"). It
features inference,[36][37][38][39][40][41] as well as the
[optimization](optimization "wikilink") concepts of
[training](training "wikilink") and
[testing](test_(assessment) "wikilink"), related to fitting and
[generalization](generalization "wikilink"), respectively. More
specifically, the probabilistic interpretation considers the activation
nonlinearity as a [cumulative distribution
function](cumulative_distribution_function "wikilink").[42] The
probabilistic interpretation led to the introduction of
[dropout](dropout_(neural_networks) "wikilink") as
[regularizer](Regularization_(mathematics) "wikilink") in neural
networks. The probabilistic interpretation was introduced by researchers
including [Hopfield](John_Hopfield "wikilink"),
[Widrow](Bernard_Widrow "wikilink") and
[Narendra](Kumpati_S._Narendra "wikilink") and popularized in surveys
such as the one by [Bishop](Christopher_Bishop "wikilink").[43]

## History

There were two [types](Types_of_artificial_neural_networks "wikilink")
of artificial neural network (ANN): [feedforward neural
networks](feedforward_neural_networks "wikilink") (FNNs) and [recurrent
neural networks](recurrent_neural_networks "wikilink") (RNNs). RNNs have
cycles in their connectivity structure, FNNs don't. In the 1920s,
[Wilhelm Lenz](Wilhelm_Lenz "wikilink") and [Ernst
Ising](Ernst_Ising "wikilink") created and analyzed the [Ising
model](Ising_model "wikilink")[44] which is essentially a non-learning
RNN architecture consisting of neuron-like threshold elements. In 1972,
[Shun'ichi Amari](Shun'ichi_Amari "wikilink") made this architecture
adaptive.[45][46] His learning RNN was popularised by [John
Hopfield](John_Hopfield "wikilink") in 1982.[47]

Charles Tappert writes that [Frank
Rosenblatt](Frank_Rosenblatt "wikilink") developed and explored all of
the basic ingredients of the deep learning systems of today,[48]
referring to Rosenblatt's 1962 book[49] which introduced [multilayer
perceptron](multilayer_perceptron "wikilink") (MLP) with 3 layers: an
input layer, a hidden layer with randomized weights that did not learn,
and an output layer. It also introduced variants, including a version
with four-layer perceptrons where the last two layers have learned
weights (and thus a proper multilayer perceptron).[50] In addition, term
deep learning was proposed in 1986 by [Rina
Dechter](Rina_Dechter "wikilink")[51] although the history of its
appearance is apparently more complicated.[52]

The first general, working learning algorithm for supervised, deep,
feedforward, multilayer [perceptrons](perceptron "wikilink") was
published by [Alexey Ivakhnenko](Alexey_Ivakhnenko "wikilink") and Lapa
in 1967.[53] A 1971 paper described a deep network with eight layers
trained by the [group method of data
handling](group_method_of_data_handling "wikilink").[54]

The first deep learning [multilayer
perceptron](multilayer_perceptron "wikilink") trained by [stochastic
gradient descent](stochastic_gradient_descent "wikilink")[55] was
published in 1967 by [Shun'ichi
Amari](Shun'ichi_Amari "wikilink").[56][57] In computer experiments
conducted by Amari's student Saito, a five layer MLP with two modifiable
layers learned [internal
representations](Knowledge_representation "wikilink") to classify
non-linearily separable pattern classes.[58] In 1987 Matthew Brand
reported that wide 12-layer nonlinear perceptrons could be fully
end-to-end trained to reproduce logic functions of nontrivial circuit
depth via gradient descent on small batches of random input/output
samples, but concluded that training time on contemporary hardware
(sub-megaflop computers) made the technique impractical, and proposed
using fixed random early layers as an input hash for a single modifiable
layer.[59] Instead, subsequent developments in hardware and
hyperparameter tunings have made end-to-end [stochastic gradient
descent](stochastic_gradient_descent "wikilink") the currently dominant
training technique.

In 1970, [Seppo Linnainmaa](Seppo_Linnainmaa "wikilink") published the
reverse mode of [automatic
differentiation](automatic_differentiation "wikilink") of discrete
connected networks of nested
[differentiable](Differentiable_function "wikilink")
functions.[60][61][62] This became known as
[backpropagation](backpropagation "wikilink").[63] It is an efficient
application of the [chain rule](chain_rule "wikilink") derived by
[Gottfried Wilhelm Leibniz](Gottfried_Wilhelm_Leibniz "wikilink") in
1673[64] to networks of differentiable nodes.[65] The terminology
"back-propagating errors" was actually introduced in 1962 by
Rosenblatt,[66][67] but he did not know how to implement this, although
[Henry J. Kelley](Henry_J._Kelley "wikilink") had a continuous precursor
of backpropagation[68] already in 1960 in the context of [control
theory](control_theory "wikilink").[69] In 1982, [Paul
Werbos](Paul_Werbos "wikilink") applied backpropagation to MLPs in the
way that has become standard.[70][71][72] In 1985, [David E.
Rumelhart](David_E._Rumelhart "wikilink") et al. published an
experimental analysis of the technique.[73]

Deep learning architectures for [convolutional neural
networks](convolutional_neural_network "wikilink") (CNNs) with
convolutional layers and downsampling layers began with the
[Neocognitron](Neocognitron "wikilink") introduced by [Kunihiko
Fukushima](Kunihiko_Fukushima "wikilink") in 1980.[74] In 1969, he also
introduced the [ReLU](rectifier_(neural_networks) "wikilink") (rectified
linear unit) [activation
function](activation_function "wikilink").[75][76] The rectifier has
become the most popular activation function for CNNs and deep learning
in general.[77] CNNs have become an essential tool for [computer
vision](computer_vision "wikilink").

The term *Deep Learning* was introduced to the machine learning
community by [Rina Dechter](Rina_Dechter "wikilink") in 1986,[78] and to
artificial neural networks by Igor Aizenberg and colleagues in 2000, in
the context of [Boolean](Boolean_network "wikilink") threshold
neurons.[79][80]

In 1988, Wei Zhang et al. applied the backpropagation algorithm to a
convolutional neural network (a simplified Neocognitron with
convolutional interconnections between the image feature layers and the
last fully connected layer) for alphabet recognition. They also proposed
an implementation of the CNN with an optical computing system.[81][82]
In 1989, [Yann LeCun](Yann_LeCun "wikilink") et al. applied
backpropagation to a CNN with the purpose of [recognizing handwritten
ZIP codes](Handwriting_recognition "wikilink") on mail. While the
algorithm worked, training required 3 days.[83] Subsequently, Wei Zhang,
et al. modified their model by removing the last fully connected layer
and applied it for medical image object segmentation in 1991[84] and
breast cancer detection in mammograms in 1994.[85] LeNet-5 (1998), a
7-level CNN by [Yann LeCun](Yann_LeCun "wikilink") et al.,[86] that
classifies digits, was applied by several banks to recognize
hand-written numbers on checks digitized in 32x32 pixel images.

In the 1980s, backpropagation did not work well for deep learning with
long credit assignment paths. To overcome this problem, [Jürgen
Schmidhuber](Jürgen_Schmidhuber "wikilink") (1992) proposed a hierarchy
of RNNs pre-trained one level at a time by [self-supervised
learning](self-supervised_learning "wikilink").[87] It uses [predictive
coding](predictive_coding "wikilink") to learn [internal
representations](Knowledge_representation "wikilink") at multiple
self-organizing time scales. This can substantially facilitate
downstream deep learning. The RNN hierarchy can be *collapsed* into a
single RNN, by [distilling](Knowledge_distillation "wikilink") a higher
level *chunker* network into a lower level *automatizer*
network.[88][89] In 1993, a chunker solved a deep learning task whose
depth exceeded 1000.[90]

In 1992, [Jürgen Schmidhuber](Jürgen_Schmidhuber "wikilink") also
published an *alternative to RNNs*[91] which is now called a *linear
[Transformer](Transformer_(machine_learning_model) "wikilink")* or a
Transformer with linearized
[self-attention](Attention_(machine_learning) "wikilink")[92][93][94]
(save for a normalization operator). It learns *internal spotlights of
attention*:[95] a slow [feedforward neural
network](feedforward_neural_network "wikilink") learns by [gradient
descent](gradient_descent "wikilink") to control the fast weights of
another neural network through [outer
products](outer_product "wikilink") of self-generated activation
patterns *FROM* and *TO* (which are now called *key* and *value* for
[self-attention](Attention_(machine_learning) "wikilink")).[96] This
fast weight *attention mapping* is applied to a query pattern.

The modern
[Transformer](Transformer_(machine_learning_model) "wikilink") was
introduced by Ashish Vaswani et al. in their 2017 paper "Attention Is
All You Need".[97] It combines this with a [softmax](softmax "wikilink")
operator and a projection matrix.[98] Transformers have increasingly
become the model of choice for [natural language
processing](natural_language_processing "wikilink").[99] Many modern
large language models such as [ChatGPT](ChatGPT "wikilink"),
[GPT-4](GPT-4 "wikilink"), and [BERT](BERT_(language_model) "wikilink")
use it. Transformers are also increasingly being used in [computer
vision](computer_vision "wikilink").[100]

In 1991, [Jürgen Schmidhuber](Jürgen_Schmidhuber "wikilink") also
published adversarial neural networks that contest with each other in
the form of a [zero-sum game](zero-sum_game "wikilink"), where one
network's gain is the other network's loss.[101][102][103] The first
network is a [generative model](generative_model "wikilink") that models
a [probability distribution](probability_distribution "wikilink") over
output patterns. The second network learns by [gradient
descent](gradient_descent "wikilink") to predict the reactions of the
environment to these patterns. This was called "artificial curiosity".
In 2014, this principle was used in a [generative adversarial
network](generative_adversarial_network "wikilink") (GAN) by [Ian
Goodfellow](Ian_Goodfellow "wikilink") et al.[104] Here the
environmental reaction is 1 or 0 depending on whether the first
network's output is in a given set. This can be used to create realistic
[deepfakes](deepfake "wikilink").[105] Excellent image quality is
achieved by [Nvidia](Nvidia "wikilink")'s
[StyleGAN](StyleGAN "wikilink") (2018)[106] based on the Progressive GAN
by Tero Karras et al.[107] Here the GAN generator is grown from small to
large scale in a pyramidal fashion.

[Sepp Hochreiter](Sepp_Hochreiter "wikilink")'s diploma thesis
(1991)[108] was called "one of the most important documents in the
history of machine learning" by his supervisor
[Schmidhuber](Jürgen_Schmidhuber "wikilink").[109] It not only tested
the neural history compressor,[110] but also identified and analyzed the
[vanishing gradient
problem](vanishing_gradient_problem "wikilink").[111][112] Hochreiter
proposed recurrent [residual](Residual_neural_network "wikilink")
connections to solve this problem. This led to the deep learning method
called [long short-term memory](long_short-term_memory "wikilink")
(LSTM), published in 1997.[113] LSTM recurrent neural networks can learn
"very deep learning" tasks[114] with long credit assignment paths that
require memories of events that happened thousands of discrete time
steps before. The "vanilla LSTM" with forget gate was introduced in 1999
by [Felix Gers](Felix_Gers "wikilink"),
[Schmidhuber](Jürgen_Schmidhuber "wikilink") and Fred Cummins.[115]
[LSTM](LSTM "wikilink") has become the most cited neural network of the
20th century.[116] In 2015, Rupesh Kumar Srivastava, Klaus Greff, and
Schmidhuber used [LSTM](LSTM "wikilink") principles to create the
[Highway network](Highway_network "wikilink"), a [feedforward neural
network](feedforward_neural_network "wikilink") with hundreds of layers,
much deeper than previous networks.[117][118] 7 months later, Kaiming
He, Xiangyu Zhang; Shaoqing Ren, and Jian Sun won the ImageNet 2015
competition with an open-gated or gateless [Highway
network](Highway_network "wikilink") variant called [Residual neural
network](Residual_neural_network "wikilink").[119] This has become the
most cited neural network of the 21st century.[120]

In 1994, André de Carvalho, together with Mike Fairhurst and David
Bisset, published experimental results of a multi-layer boolean neural
network, also known as a weightless neural network, composed of a
3-layers self-organising feature extraction neural network module (SOFT)
followed by a multi-layer classification neural network module (GSN),
which were independently trained. Each layer in the feature extraction
module extracted features with growing complexity regarding the previous
layer.[121]

In 1995, [Brendan Frey](Brendan_Frey "wikilink") demonstrated that it
was possible to train (over two days) a network containing six fully
connected layers and several hundred hidden units using the [wake-sleep
algorithm](wake-sleep_algorithm "wikilink"), co-developed with [Peter
Dayan](Peter_Dayan "wikilink") and
[Hinton](Geoffrey_Hinton "wikilink").[122]

Since 1997, Sven Behnke extended the feed-forward hierarchical
convolutional approach in the Neural Abstraction Pyramid[123] by lateral
and backward connections in order to flexibly incorporate context into
decisions and iteratively resolve local ambiguities.

Simpler models that use task-specific handcrafted features such as
[Gabor filters](Gabor_filter "wikilink") and [support vector
machines](support_vector_machine "wikilink") (SVMs) were a popular
choice in the 1990s and 2000s, because of artificial neural networks'
computational cost and a lack of understanding of how the brain wires
its biological networks.

Both shallow and deep learning (e.g., recurrent nets) of ANNs for
[speech recognition](speech_recognition "wikilink") have been explored
for many years.[124][125][126] These methods never outperformed
non-uniform internal-handcrafting Gaussian [mixture
model](mixture_model "wikilink")/[Hidden Markov
model](Hidden_Markov_model "wikilink") (GMM-HMM) technology based on
generative models of speech trained discriminatively.[127] Key
difficulties have been analyzed, including gradient diminishing[128] and
weak temporal correlation structure in neural predictive
models.[129][130] Additional difficulties were the lack of training data
and limited computing power. Most [speech
recognition](speech_recognition "wikilink") researchers moved away from
neural nets to pursue generative modeling. An exception was at [SRI
International](SRI_International "wikilink") in the late 1990s. Funded
by the US government's [NSA](National_Security_Agency "wikilink") and
[DARPA](DARPA "wikilink"), SRI studied deep neural networks (DNNs) in
speech and [speaker recognition](speaker_recognition "wikilink"). The
speaker recognition team led by [Larry Heck](Larry_Heck "wikilink")
reported significant success with deep neural networks in speech
processing in the 1998 [National Institute of Standards and
Technology](National_Institute_of_Standards_and_Technology "wikilink")
Speaker Recognition evaluation.[131] The SRI deep neural network was
then deployed in the Nuance Verifier, representing the first major
industrial application of deep learning.[132] The principle of elevating
"raw" features over hand-crafted optimization was first explored
successfully in the architecture of deep autoencoder on the "raw"
spectrogram or linear [filter-bank](Filter_bank "wikilink") features in
the late 1990s,[133] showing its superiority over the
[Mel-Cepstral](Mel-frequency_cepstrum "wikilink") features that contain
stages of fixed transformation from spectrograms. The raw features of
speech, [waveforms](waveform "wikilink"), later produced excellent
larger-scale results.[134]

Speech recognition was taken over by [LSTM](LSTM "wikilink"). In 2003,
LSTM started to become competitive with traditional speech recognizers
on certain tasks.[135] In 2006, [Alex
Graves](Alex_Graves_(computer_scientist) "wikilink"), Santiago
Fernández, Faustino Gomez, and Schmidhuber combined it with
[connectionist temporal
classification](connectionist_temporal_classification "wikilink")
(CTC)[136] in stacks of LSTM RNNs.[137] In 2015, Google's speech
recognition reportedly experienced a dramatic performance jump of 49%
through CTC-trained LSTM, which they made available through [Google
Voice Search](Google_Voice_Search "wikilink").[138]

The impact of deep learning in industry began in the early 2000s, when
CNNs already processed an estimated 10% to 20% of all the checks written
in the US, according to Yann LeCun.[139] Industrial applications of deep
learning to large-scale speech recognition started around 2010.

In 2006, publications by [Geoff Hinton](Geoffrey_Hinton "wikilink"),
[Ruslan Salakhutdinov](Russ_Salakhutdinov "wikilink"), Osindero and
[Teh](Yee_Whye_Teh "wikilink")[140][141][142] showed how a many-layered
[feedforward neural network](feedforward_neural_network "wikilink")
could be effectively pre-trained one layer at a time, treating each
layer in turn as an unsupervised [restricted Boltzmann
machine](restricted_Boltzmann_machine "wikilink"), then
[fine-tuning](Fine-tuning_(deep_learning) "wikilink") it using
supervised backpropagation.[143] The papers referred to *learning* for
*deep belief nets.*

The 2009 NIPS Workshop on Deep Learning for Speech Recognition was
motivated by the limitations of deep generative models of speech, and
the possibility that given more capable hardware and large-scale data
sets that deep neural nets might become practical. It was believed that
pre-training DNNs using generative models of deep belief nets (DBN)
would overcome the main difficulties of neural nets. However, it was
discovered that replacing pre-training with large amounts of training
data for straightforward backpropagation when using DNNs with large,
context-dependent output layers produced error rates dramatically lower
than then-state-of-the-art Gaussian mixture model (GMM)/Hidden Markov
Model (HMM) and also than more-advanced generative model-based
systems.[144] The nature of the recognition errors produced by the two
types of systems was characteristically different,[145] offering
technical insights into how to integrate deep learning into the existing
highly efficient, run-time speech decoding system deployed by all major
speech recognition systems.[146][147][148] Analysis around 2009–2010,
contrasting the GMM (and other generative speech models) vs. DNN models,
stimulated early industrial investment in deep learning for speech
recognition.[149] That analysis was done with comparable performance
(less than 1.5% in error rate) between discriminative DNNs and
generative models.[150][151][152] In 2010, researchers extended deep
learning from [TIMIT](TIMIT "wikilink") to large vocabulary speech
recognition, by adopting large output layers of the DNN based on
context-dependent HMM states constructed by [decision
trees](decision_tree "wikilink").[153][154][155][156]

Deep learning is part of state-of-the-art systems in various
disciplines, particularly computer vision and [automatic speech
recognition](automatic_speech_recognition "wikilink") (ASR). Results on
commonly used evaluation sets such as [TIMIT](TIMIT "wikilink") (ASR)
and [MNIST](MNIST_database "wikilink") ([image
classification](image_classification "wikilink")), as well as a range of
large-vocabulary speech recognition tasks have steadily
improved.[157][158] Convolutional neural networks were superseded for
ASR by CTC[159] for [LSTM](LSTM "wikilink").[160][161][162][163][164]
but are more successful in computer vision.

Advances in hardware have driven renewed interest in deep learning. In
2009, [Nvidia](Nvidia "wikilink") was involved in what was called the
"big bang" of deep learning, "as deep-learning neural networks were
trained with Nvidia [graphics processing
units](graphics_processing_unit "wikilink") (GPUs)".[165] That year,
[Andrew Ng](Andrew_Ng "wikilink") determined that GPUs could increase
the speed of deep-learning systems by about 100 times.[166] In
particular, GPUs are well-suited for the matrix/vector computations
involved in machine learning.[167][168][169] GPUs speed up training
algorithms by orders of magnitude, reducing running times from weeks to
days.[170][171] Further, specialized hardware and algorithm
optimizations can be used for efficient processing of deep learning
models.[172]

### Deep learning revolution

<figure>
<img src="AI-ML-DL.svg"
title="How deep learning is a subset of machine learning and how machine learning is a subset of artificial intelligence (AI)" />
<figcaption>How deep learning is a subset of machine learning and how
machine learning is a subset of artificial intelligence
(AI)</figcaption>
</figure>

In the late 2000s, deep learning started to outperform other methods in
machine learning competitions. In 2009, a [long short-term
memory](long_short-term_memory "wikilink") trained by [connectionist
temporal
classification](connectionist_temporal_classification "wikilink") ([Alex
Graves](Alex_Graves_(computer_scientist) "wikilink"), Santiago
Fernández, Faustino Gomez, and [Jürgen
Schmidhuber](Jürgen_Schmidhuber "wikilink"), 2006)[173] was the first
RNN to win [pattern recognition](pattern_recognition "wikilink")
contests, winning three competitions in connected [handwriting
recognition](handwriting_recognition "wikilink").[174][175]
[Google](Google "wikilink") later used CTC-trained LSTM for speech
recognition on the [smartphone](smartphone "wikilink").[176][177]

Significant impacts in image or object recognition were felt from 2011
to 2012. Although CNNs trained by backpropagation had been around for
decades,[178][179] and GPU implementations of NNs for years,[180]
including CNNs,[181][182] faster implementations of CNNs on GPUs were
needed to progress on computer vision. In 2011, the *DanNet*[183][184]
by Dan Ciresan, Ueli Meier, Jonathan Masci, [Luca Maria
Gambardella](Luca_Maria_Gambardella "wikilink"), and [Jürgen
Schmidhuber](Jürgen_Schmidhuber "wikilink") achieved for the first time
superhuman performance in a visual pattern recognition contest,
outperforming traditional methods by a factor of 3.[185] Also in 2011,
DanNet won the ICDAR Chinese handwriting contest, and in May 2012, it
won the ISBI image segmentation contest.[186] Until 2011, CNNs did not
play a major role at computer vision conferences, but in June 2012, a
paper by Ciresan et al. at the leading conference CVPR[187] showed how
[max-pooling](Max_pooling "wikilink") CNNs on GPU can dramatically
improve many vision benchmark records. In September 2012, DanNet also
won the ICPR contest on analysis of large medical images for cancer
detection, and in the following year also the MICCAI Grand Challenge on
the same topic.[188] In October 2012, the similar
[AlexNet](AlexNet "wikilink") by [Alex
Krizhevsky](Alex_Krizhevsky "wikilink"), [Ilya
Sutskever](Ilya_Sutskever "wikilink"), and [Geoffrey
Hinton](Geoffrey_Hinton "wikilink")[189] won the large-scale [ImageNet
competition](ImageNet_competition "wikilink") by a significant margin
over shallow machine learning methods. The VGG-16 network by [Karen
Simonyan](Karen_Simonyan "wikilink") and [Andrew
Zisserman](Andrew_Zisserman "wikilink")[190] further reduced the error
rate and won the ImageNet 2014 competition, following a similar trend in
large-scale speech recognition.

Image classification was then extended to the more challenging task of
[generating descriptions](Automatic_image_annotation "wikilink")
(captions) for images, often as a combination of CNNs and
LSTMs.[191][192][193]

In 2012, a team led by George E. Dahl won the "Merck Molecular Activity
Challenge" using multi-task deep neural networks to predict the
[biomolecular target](biomolecular_target "wikilink") of one
drug.[194][195] In 2014, [Sepp Hochreiter](Sepp_Hochreiter "wikilink")'s
group used deep learning to detect off-target and toxic effects of
environmental chemicals in nutrients, household products and drugs and
won the "Tox21 Data Challenge" of [NIH](NIH "wikilink"),
[FDA](FDA "wikilink") and
[NCATS](National_Center_for_Advancing_Translational_Sciences "wikilink").[196][197][198]

In 2016, Roger Parloff mentioned a "deep learning revolution" that has
transformed the AI industry.[199]

In March 2019, [Yoshua Bengio](Yoshua_Bengio "wikilink"), [Geoffrey
Hinton](Geoffrey_Hinton "wikilink") and [Yann
LeCun](Yann_LeCun "wikilink") were awarded the [Turing
Award](Turing_Award "wikilink") for conceptual and engineering
breakthroughs that have made deep neural networks a critical component
of computing.

## Neural networks

**Artificial neural networks** (**ANNs**) or
**[connectionist](Connectionism "wikilink") systems** are computing
systems inspired by the [biological neural
networks](biological_neural_network "wikilink") that constitute animal
brains. Such systems learn (progressively improve their ability) to do
tasks by considering examples, generally without task-specific
programming. For example, in image recognition, they might learn to
identify images that contain cats by analyzing example images that have
been manually [labeled](Labeled_data "wikilink") as "cat" or "no cat"
and using the analytic results to identify cats in other images. They
have found most use in applications difficult to express with a
traditional computer algorithm using [rule-based
programming](rule-based_programming "wikilink").

An ANN is based on a collection of connected units called [artificial
neurons](artificial_neuron "wikilink"), (analogous to biological
[neurons](neuron "wikilink") in a [biological brain](Brain "wikilink")).
Each connection ([synapse](synapse "wikilink")) between neurons can
transmit a signal to another neuron. The receiving (postsynaptic) neuron
can process the signal(s) and then signal downstream neurons connected
to it. Neurons may have state, generally represented by [real
numbers](real_numbers "wikilink"), typically between 0 and 1. Neurons
and synapses may also have a weight that varies as learning proceeds,
which can increase or decrease the strength of the signal that it sends
downstream.

Typically, neurons are organized in layers. Different layers may perform
different kinds of transformations on their inputs. Signals travel from
the first (input), to the last (output) layer, possibly after traversing
the layers multiple times.

The original goal of the neural network approach was to solve problems
in the same way that a human brain would. Over time, attention focused
on matching specific mental abilities, leading to deviations from
biology such as [backpropagation](backpropagation "wikilink"), or
passing information in the reverse direction and adjusting the network
to reflect that information.

Neural networks have been used on a variety of tasks, including computer
vision, [speech recognition](speech_recognition "wikilink"), [machine
translation](machine_translation "wikilink"), [social
network](social_network "wikilink") filtering, [playing board and video
games](general_game_playing "wikilink") and medical diagnosis.

As of 2017, neural networks typically have a few thousand to a few
million units and millions of connections. Despite this number being
several order of magnitude less than the number of neurons on a human
brain, these networks can perform many tasks at a level beyond that of
humans (e.g., recognizing faces, or playing "Go"[200]).

### Deep neural networks

A deep neural network (DNN) is an artificial neural network with
multiple layers between the input and output layers.[201][202] There are
different types of neural networks but they always consist of the same
components: neurons, synapses, weights, biases, and functions.[203]
These components as a whole function in a way that mimics functions of
the human brain, and can be trained like any other ML algorithm.

For example, a DNN that is trained to recognize dog breeds will go over
the given image and calculate the probability that the dog in the image
is a certain breed. The user can review the results and select which
probabilities the network should display (above a certain threshold,
etc.) and return the proposed label. Each mathematical manipulation as
such is considered a layer, and complex DNN have many layers, hence the
name "deep" networks.

DNNs can model complex non-linear relationships. DNN architectures
generate compositional models where the object is expressed as a layered
composition of [primitives](Primitive_data_type "wikilink").[204] The
extra layers enable composition of features from lower layers,
potentially modeling complex data with fewer units than a similarly
performing shallow network.[205] For instance, it was proved that sparse
[multivariate polynomials](multivariate_polynomial "wikilink") are
exponentially easier to approximate with DNNs than with shallow
networks.[206]

Deep architectures include many variants of a few basic approaches. Each
architecture has found success in specific domains. It is not always
possible to compare the performance of multiple architectures, unless
they have been evaluated on the same data sets.

DNNs are typically feedforward networks in which data flows from the
input layer to the output layer without looping back. At first, the DNN
creates a map of virtual neurons and assigns random numerical values, or
"weights", to connections between them. The weights and inputs are
multiplied and return an output between 0 and 1. If the network did not
accurately recognize a particular pattern, an algorithm would adjust the
weights.[207] That way the algorithm can make certain parameters more
influential, until it determines the correct mathematical manipulation
to fully process the data.

[Recurrent neural networks](Recurrent_neural_networks "wikilink"), in
which data can flow in any direction, are used for applications such as
[language modeling](language_model "wikilink").[208][209][210][211][212]
Long short-term memory is particularly effective for this use.[213][214]

[Convolutional neural networks](Convolutional_neural_network "wikilink")
(CNNs) are used in computer vision.[215] CNNs also have been applied to
[acoustic modeling](acoustic_model "wikilink") for automatic speech
recognition (ASR).[216]

#### Challenges

As with ANNs, many issues can arise with naively trained DNNs. Two
common issues are [overfitting](overfitting "wikilink") and computation
time.

DNNs are prone to overfitting because of the added layers of
abstraction, which allow them to model rare dependencies in the training
data. [Regularization](Regularization_(mathematics) "wikilink") methods
such as Ivakhnenko's unit pruning[217] or [weight
decay](weight_decay "wikilink") (ℓ<sub>2</sub>-regularization) or
[sparsity](sparse_matrix "wikilink") (ℓ<sub>1</sub>-regularization) can
be applied during training to combat overfitting.[218] Alternatively
[dropout](Dropout_(neural_networks) "wikilink") regularization randomly
omits units from the hidden layers during training. This helps to
exclude rare dependencies.[219] Finally, data can be augmented via
methods such as cropping and rotating such that smaller training sets
can be increased in size to reduce the chances of overfitting.[220]

DNNs must consider many training parameters, such as the size (number of
layers and number of units per layer), the [learning
rate](learning_rate "wikilink"), and initial weights. [Sweeping through
the parameter space](Hyperparameter_optimization#Grid_search "wikilink")
for optimal parameters may not be feasible due to the cost in time and
computational resources. Various tricks, such as
[batching](Batch_learning "wikilink") (computing the gradient on several
training examples at once rather than individual examples)[221] speed up
computation. Large processing capabilities of many-core architectures
(such as GPUs or the Intel Xeon Phi) have produced significant speedups
in training, because of the suitability of such processing architectures
for the matrix and vector computations.[222][223]

Alternatively, engineers may look for other types of neural networks
with more straightforward and convergent training algorithms. CMAC
([cerebellar model articulation
controller](cerebellar_model_articulation_controller "wikilink")) is one
such kind of neural network. It doesn't require learning rates or
randomized initial weights. The training process can be guaranteed to
converge in one step with a new batch of data, and the computational
complexity of the training algorithm is linear with respect to the
number of neurons involved.[224][225]

## Hardware

Since the 2010s, advances in both machine learning algorithms and
[computer hardware](computer_hardware "wikilink") have led to more
efficient methods for training deep neural networks that contain many
layers of non-linear hidden units and a very large output layer.[226] By
2019, graphic processing units ([GPUs](GPU "wikilink")), often with
AI-specific enhancements, had displaced CPUs as the dominant method of
training large-scale commercial cloud AI.[227]
[OpenAI](OpenAI "wikilink") estimated the hardware computation used in
the largest deep learning projects from AlexNet (2012) to AlphaZero
(2017), and found a 300,000-fold increase in the amount of computation
required, with a doubling-time trendline of 3.4 months.[228][229]

Special [electronic circuits](electronic_circuit "wikilink") called
[deep learning processors](deep_learning_processor "wikilink") were
designed to speed up deep learning algorithms. Deep learning processors
include neural processing units (NPUs) in [Huawei](Huawei "wikilink")
cellphones[230] and [cloud computing](cloud_computing "wikilink")
servers such as [tensor processing
units](tensor_processing_unit "wikilink") (TPU) in the [Google Cloud
Platform](Google_Cloud_Platform "wikilink").[231] [Cerebras
Systems](Cerebras "wikilink") has also built a dedicated system to
handle large deep learning models, the CS-2, based on the largest
processor in the industry, the second-generation Wafer Scale Engine
(WSE-2).[232][233]

Atomically thin [semiconductors](semiconductors "wikilink") are
considered promising for energy-efficient deep learning hardware where
the same basic device structure is used for both logic operations and
data storage. In 2020, Marega et al. published experiments with a
large-area active channel material for developing logic-in-memory
devices and circuits based on [floating-gate](floating-gate "wikilink")
[field-effect transistors](field-effect_transistor "wikilink")
(FGFETs).[234]

In 2021, J. Feldmann et al. proposed an integrated
[photonic](photonic "wikilink") [hardware
accelerator](hardware_accelerator "wikilink") for parallel convolutional
processing.[235] The authors identify two key advantages of integrated
photonics over its electronic counterparts: (1) massively parallel data
transfer through [wavelength](wavelength "wikilink") division
[multiplexing](multiplexing "wikilink") in conjunction with [frequency
combs](frequency_comb "wikilink"), and (2) extremely high data
modulation speeds.[236] Their system can execute trillions of
multiply-accumulate operations per second, indicating the potential of
[integrated](Photonic_integrated_circuit "wikilink")
[photonics](photonics "wikilink") in data-heavy AI applications.[237]

## Applications

### Automatic speech recognition

Large-scale automatic speech recognition is the first and most
convincing successful case of deep learning. LSTM RNNs can learn "Very
Deep Learning" tasks[238] that involve multi-second intervals containing
speech events separated by thousands of discrete time steps, where one
time step corresponds to about 10 ms. LSTM with forget gates[239] is
competitive with traditional speech recognizers on certain tasks.[240]

The initial success in speech recognition was based on small-scale
recognition tasks based on TIMIT. The data set contains 630 speakers
from eight major [dialects](dialect "wikilink") of [American
English](American_English "wikilink"), where each speaker reads 10
sentences.[241] Its small size lets many configurations be tried. More
importantly, the TIMIT task concerns
[phone](Phone_(phonetics) "wikilink")-sequence recognition, which,
unlike word-sequence recognition, allows weak phone
[bigram](bigram "wikilink") language models. This lets the strength of
the acoustic modeling aspects of speech recognition be more easily
analyzed. The error rates listed below, including these early results
and measured as percent phone error rates (PER), have been summarized
since 1991.

<table>
<thead>
<tr class="header">
<th><p>Method</p></th>
<th><p>Percent phone<br />
error rate (PER) (%)</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><p>Randomly Initialized RNN<a href="#fn1" class="footnote-ref"
id="fnref1" role="doc-noteref"><sup>1</sup></a></p></td>
<td><p>26.1</p></td>
</tr>
<tr class="even">
<td><p>Bayesian Triphone GMM-HMM</p></td>
<td><p>25.6</p></td>
</tr>
<tr class="odd">
<td><p>Hidden Trajectory (Generative) Model</p></td>
<td><p>24.8</p></td>
</tr>
<tr class="even">
<td><p>Monophone Randomly Initialized DNN</p></td>
<td><p>23.4</p></td>
</tr>
<tr class="odd">
<td><p>Monophone DBN-DNN</p></td>
<td><p>22.4</p></td>
</tr>
<tr class="even">
<td><p>Triphone GMM-HMM with BMMI Training</p></td>
<td><p>21.7</p></td>
</tr>
<tr class="odd">
<td><p>Monophone DBN-DNN on fbank</p></td>
<td><p>20.7</p></td>
</tr>
<tr class="even">
<td><p>Convolutional DNN<a href="#fn2" class="footnote-ref" id="fnref2"
role="doc-noteref"><sup>2</sup></a></p></td>
<td><p>20.0</p></td>
</tr>
<tr class="odd">
<td><p>Convolutional DNN w. Heterogeneous Pooling</p></td>
<td><p>18.7</p></td>
</tr>
<tr class="even">
<td><p>Ensemble DNN/CNN/RNN<a href="#fn3" class="footnote-ref"
id="fnref3" role="doc-noteref"><sup>3</sup></a></p></td>
<td><p>18.3</p></td>
</tr>
<tr class="odd">
<td><p>Bidirectional LSTM</p></td>
<td><p>17.8</p></td>
</tr>
<tr class="even">
<td><p>Hierarchical Convolutional Deep Maxout Network<a href="#fn4"
class="footnote-ref" id="fnref4"
role="doc-noteref"><sup>4</sup></a></p></td>
<td><p>16.5</p></td>
</tr>
</tbody>
</table>
<section id="footnotes" class="footnotes footnotes-end-of-document"
role="doc-endnotes">
<hr />
<ol>
<li id="fn1"><a href="#fnref1" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn2"><a href="#fnref2" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn3"><a href="#fnref3" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn4"><a href="#fnref4" class="footnote-back"
role="doc-backlink">↩︎</a></li>
</ol>
</section>

The debut of DNNs for speaker recognition in the late 1990s and speech
recognition around 2009-2011 and of LSTM around 2003–2007, accelerated
progress in eight major areas:[242][243][244]

-   Scale-up/out and accelerated DNN training and decoding
-   Sequence discriminative training
-   Feature processing by deep models with solid understanding of the
    underlying mechanisms
-   Adaptation of DNNs and related deep models
-   [Multi-task](Multi-task_learning "wikilink") and [transfer
    learning](transfer_learning "wikilink") by DNNs and related deep
    models
-   [CNNs](Convolutional_neural_network "wikilink") and how to design
    them to best exploit [domain knowledge](domain_knowledge "wikilink")
    of speech
-   [RNN](Recurrent_neural_network "wikilink") and its rich LSTM
    variants
-   Other types of deep models including tensor-based models and
    integrated deep generative/discriminative models.

All major commercial speech recognition systems (e.g., Microsoft
[Cortana](Cortana_(software) "wikilink"), [Xbox](Xbox "wikilink"),
[Skype Translator](Skype_Translator "wikilink"), [Amazon
Alexa](Amazon_Alexa "wikilink"), [Google Now](Google_Now "wikilink"),
[Apple Siri](Siri "wikilink"), [Baidu](Baidu "wikilink") and
[iFlyTek](IFlytek "wikilink") voice search, and a range of
[Nuance](Nuance_Communications "wikilink") speech products, etc.) are
based on deep learning.[245][246][247]

### Image recognition

A common evaluation set for image classification is the [MNIST
database](MNIST_database "wikilink") data set. MNIST is composed of
handwritten digits and includes 60,000 training examples and 10,000 test
examples. As with TIMIT, its small size lets users test multiple
configurations. A comprehensive list of results on this set is
available.[248]

Deep learning-based image recognition has become "superhuman", producing
more accurate results than human contestants. This first occurred in
2011 in recognition of traffic signs, and in 2014, with recognition of
human faces.[249][250]

Deep learning-trained vehicles now interpret 360° camera views.[251]
Another example is Facial Dysmorphology Novel Analysis (FDNA) used to
analyze cases of human malformation connected to a large database of
genetic syndromes.

### Visual art processing

<img
src="Jimmy_Wales_in_France,_with_the_style_of_Munch&#39;s_&quot;The_Scream&quot;_applied_using_neural_style_transfer.jpg"
title="Visual art processing of Jimmy Wales in France, with the style of Munch&#39;s &quot;The Scream&quot; applied using neural style transfer"
width="164" height="164"
alt="Visual art processing of Jimmy Wales in France, with the style of Munch&#39;s &quot;The Scream&quot; applied using neural style transfer" />
Closely related to the progress that has been made in image recognition
is the increasing application of deep learning techniques to various
visual art tasks. DNNs have proven themselves capable, for example, of

-   identifying the style period of a given painting[252][253]
-   [Neural Style Transfer](Neural_Style_Transfer "wikilink") capturing
    the style of a given artwork and applying it in a visually pleasing
    manner to an arbitrary photograph or video[254][255]
-   generating striking imagery based on random visual input
    fields.[256][257]

### Natural language processing

Neural networks have been used for implementing language models since
the early 2000s.[258] LSTM helped to improve machine translation and
language modeling.[259][260][261]

Other key techniques in this field are negative sampling[262] and [word
embedding](word_embedding "wikilink"). Word embedding, such as
*[word2vec](word2vec "wikilink")*, can be thought of as a
representational layer in a deep learning architecture that transforms
an atomic word into a positional representation of the word relative to
other words in the dataset; the position is represented as a point in a
[vector space](vector_space "wikilink"). Using word embedding as an RNN
input layer allows the network to parse sentences and phrases using an
effective compositional vector grammar. A compositional vector grammar
can be thought of as [probabilistic context free
grammar](probabilistic_context_free_grammar "wikilink") (PCFG)
implemented by an RNN.[263] Recursive auto-encoders built atop word
embeddings can assess sentence similarity and detect paraphrasing.[264]
Deep neural architectures provide the best results for [constituency
parsing](Statistical_parsing "wikilink"),[265] [sentiment
analysis](sentiment_analysis "wikilink"),[266] information
retrieval,[267][268] spoken language understanding,[269] machine
translation,[270][271] contextual entity linking,[272] writing style
recognition,[273] [named-entity
recognition](named-entity_recognition "wikilink") (token
classification),[274] text classification, and others.[275]

Recent developments generalize [word
embedding](word_embedding "wikilink") to [sentence
embedding](sentence_embedding "wikilink").

[Google Translate](Google_Translate "wikilink") (GT) uses a large
end-to-end [long short-term memory](long_short-term_memory "wikilink")
(LSTM) network.[276][277][278][279] [Google Neural Machine Translation
(GNMT)](Google_Neural_Machine_Translation "wikilink") uses an
[example-based machine
translation](example-based_machine_translation "wikilink") method in
which the system "learns from millions of examples".[280] It translates
"whole sentences at a time, rather than pieces". Google Translate
supports over one hundred languages.[281] The network encodes the
"semantics of the sentence rather than simply memorizing
phrase-to-phrase translations".[282][283] GT uses English as an
intermediate between most language pairs.[284]

### Drug discovery and toxicology

A large percentage of candidate drugs fail to win regulatory approval.
These failures are caused by insufficient efficacy (on-target effect),
undesired interactions (off-target effects), or unanticipated [toxic
effects](Toxicity "wikilink").[285][286] Research has explored use of
deep learning to predict the [biomolecular
targets](biomolecular_target "wikilink"),[287][288]
[off-targets](off-target "wikilink"), and [toxic
effects](Toxicity "wikilink") of environmental chemicals in nutrients,
household products and drugs.[289][290][291]

AtomNet is a deep learning system for structure-based [rational drug
design](Drug_design "wikilink").[292] AtomNet was used to predict novel
candidate biomolecules for disease targets such as the [Ebola
virus](Ebola_virus "wikilink")[293] and [multiple
sclerosis](multiple_sclerosis "wikilink").[294][295]

In 2017 [graph neural networks](graph_neural_network "wikilink") were
used for the first time to predict various properties of molecules in a
large toxicology data set.[296] In 2019, generative neural networks were
used to produce molecules that were validated experimentally all the way
into mice.[297][298]

### Customer relationship management

[Deep reinforcement learning](Deep_reinforcement_learning "wikilink")
has been used to approximate the value of possible [direct
marketing](direct_marketing "wikilink") actions, defined in terms of
[RFM](RFM_(customer_value) "wikilink") variables. The estimated value
function was shown to have a natural interpretation as [customer
lifetime value](customer_lifetime_value "wikilink").[299]

### Recommendation systems

Recommendation systems have used deep learning to extract meaningful
features for a latent factor model for content-based music and journal
recommendations.[300][301] Multi-view deep learning has been applied for
learning user preferences from multiple domains.[302] The model uses a
hybrid collaborative and content-based approach and enhances
recommendations in multiple tasks.

### Bioinformatics

An [autoencoder](autoencoder "wikilink") ANN was used in
[bioinformatics](bioinformatics "wikilink"), to predict [gene
ontology](Gene_Ontology "wikilink") annotations and gene-function
relationships.[303]

In medical informatics, deep learning was used to predict sleep quality
based on data from wearables[304] and predictions of health
complications from [electronic health
record](electronic_health_record "wikilink") data.[305]

Deep neural networks have shown unparalleled performance in [predicting
protein structure](Protein_structure_prediction "wikilink"), according
to the sequence of the amino acids that make it up. In 2020,
[AlphaFold](AlphaFold "wikilink"), a deep-learning based system,
achieved a level of accuracy significantly higher than all previous
computational methods.[306][307]

### Deep Neural Network Estimations

Deep neural networks can be used to estimate the entropy of a
[stochastic process](stochastic_process "wikilink") and called Neural
Joint Entropy Estimator (NJEE).[308] Such an estimation provides
insights on the effects of input [random
variables](random_variables "wikilink") on an independent [random
variable](random_variable "wikilink"). Practically, the DNN is trained
as a [classifier](Classifier_(machine_learning) "wikilink") that maps an
input [vector](Vector_(mathematics_and_physics) "wikilink") or
[matrix](Matrix_(mathematics) "wikilink") X to an output [probability
distribution](probability_distribution "wikilink") over the possible
classes of random variable Y, given input X. For example, in [image
classification](image_classification "wikilink") tasks, the NJEE maps a
vector of [pixels](pixels "wikilink")' color values to probabilities
over possible image classes. In practice, the probability distribution
of Y is obtained by a [Softmax](Softmax "wikilink") layer with number of
nodes that is equal to the [alphabet](alphabet "wikilink") size of Y.
NJEE uses continuously differentiable [activation
functions](activation_function "wikilink"), such that the conditions for
the [universal approximation
theorem](universal_approximation_theorem "wikilink") holds. It is shown
that this method provides a strongly [consistent
estimator](consistent_estimator "wikilink") and outperforms other
methods in case of large alphabet sizes.[309]

### Medical image analysis

Deep learning has been shown to produce competitive results in medical
application such as cancer cell classification, lesion detection, organ
segmentation and image enhancement.[310][311] Modern deep learning tools
demonstrate the high accuracy of detecting various diseases and the
helpfulness of their use by specialists to improve the diagnosis
efficiency.[312][313]

### Mobile advertising

Finding the appropriate mobile audience for [mobile
advertising](mobile_advertising "wikilink") is always challenging, since
many data points must be considered and analyzed before a target segment
can be created and used in ad serving by any ad server.[314] Deep
learning has been used to interpret large, many-dimensioned advertising
datasets. Many data points are collected during the request/serve/click
internet advertising cycle. This information can form the basis of
machine learning to improve ad selection.

### Image restoration

Deep learning has been successfully applied to [inverse
problems](inverse_problems "wikilink") such as
[denoising](denoising "wikilink"),
[super-resolution](super-resolution "wikilink"),
[inpainting](inpainting "wikilink"), and [film
colorization](film_colorization "wikilink").[315] These applications
include learning methods such as "Shrinkage Fields for Effective Image
Restoration"[316] which trains on an image dataset, and [Deep Image
Prior](Deep_Image_Prior "wikilink"), which trains on the image that
needs restoration.

### Financial fraud detection

Deep learning is being successfully applied to financial [fraud
detection](fraud_detection "wikilink"), tax evasion detection,[317] and
anti-money laundering.[318]

### Materials science

In November 2023, researchers at [Google
DeepMind](Google_DeepMind "wikilink") and [Lawrence Berkeley National
Laboratory](Lawrence_Berkeley_National_Laboratory "wikilink") announced
that they had developed an AI system known as GNoME. This system has
contributed to [materials science](materials_science "wikilink") by
discovering over 2 million new materials within a relatively short
timeframe. GNoME employs deep learning techniques to efficiently explore
potential material structures, achieving a significant increase in the
identification of stable inorganic [crystal
structures](crystal_structure "wikilink"). The system's predictions were
validated through autonomous robotic experiments, demonstrating a
noteworthy success rate of 71%. The data of newly discovered materials
is publicly available through the [Materials
Project](Materials_Project "wikilink") database, offering researchers
the opportunity to identify materials with desired properties for
various applications. This development has implications for the future
of scientific discovery and the integration of AI in material science
research, potentially expediting material innovation and reducing costs
in product development. The use of AI and deep learning suggests the
possibility of minimizing or eliminating manual lab experiments and
allowing scientists to focus more on the design and analysis of unique
compounds.[319][320][321]

### Military

The United States Department of Defense applied deep learning to train
robots in new tasks through observation.[322]

### Partial differential equations

Physics informed neural networks have been used to solve [partial
differential equations](partial_differential_equation "wikilink") in
both forward and inverse problems in a data driven manner.[323] One
example is the reconstructing fluid flow governed by the [Navier-Stokes
equations](Navier–Stokes_equations "wikilink"). Using physics informed
neural networks does not require the often expensive mesh generation
that conventional [CFD](Computational_fluid_dynamics "wikilink") methods
relies on.[324][325]

### Image reconstruction

Image reconstruction is the reconstruction of the underlying images from
the image-related measurements. Several works showed the better and
superior performance of the deep learning methods compared to analytical
methods for various applications, e.g., spectral imaging [326] and
ultrasound imaging.[327]

### Weather prediction

Traditional weather prediction systems solve a very complex system of
patrial differential equations. GraphCast is a deep learning based
model, trained on a long history of weather data to predict how weather
patterns change over time. It is able to predict weather conditions for
up to 10 days globally, at a very detailed level, and in under a minute,
with precision similar to state of the art systems.[328][329]

### Epigenetic clock

An epigenetic clock is a [biochemical
test](Biomarkers_of_aging "wikilink") that can be used to measure age.
Galkin et al. used deep neural networks to train an epigenetic aging
clock of unprecedented accuracy using &gt;6,000 blood samples.[330] The
clock uses information from 1000 [CpG sites](CpG_site "wikilink") and
predicts people with certain conditions older than healthy controls:
[IBD](Inflammatory_bowel_disease "wikilink"), [frontotemporal
dementia](frontotemporal_dementia "wikilink"), [ovarian
cancer](ovarian_cancer "wikilink"), [obesity](obesity "wikilink"). The
aging clock was planned to be released for public use in 2021 by an
[Insilico Medicine](Insilico_Medicine "wikilink") spinoff company Deep
Longevity.

## Relation to human cognitive and brain development

Deep learning is closely related to a class of theories of [brain
development](brain_development "wikilink") (specifically, neocortical
development) proposed by [cognitive
neuroscientists](cognitive_neuroscientist "wikilink") in the early
1990s.[331][332][333][334] These developmental theories were
instantiated in computational models, making them predecessors of deep
learning systems. These developmental models share the property that
various proposed learning dynamics in the brain (e.g., a wave of [nerve
growth factor](nerve_growth_factor "wikilink")) support the
[self-organization](self-organization "wikilink") somewhat analogous to
the neural networks utilized in deep learning models. Like the
[neocortex](neocortex "wikilink"), neural networks employ a hierarchy of
layered filters in which each layer considers information from a prior
layer (or the operating environment), and then passes its output (and
possibly the original input), to other layers. This process yields a
self-organizing stack of [transducers](transducer "wikilink"),
well-tuned to their operating environment. A 1995 description stated,
"...the infant's brain seems to organize itself under the influence of
waves of so-called trophic-factors ... different regions of the brain
become connected sequentially, with one layer of tissue maturing before
another and so on until the whole brain is mature".[335]

A variety of approaches have been used to investigate the plausibility
of deep learning models from a neurobiological perspective. On the one
hand, several variants of the
[backpropagation](backpropagation "wikilink") algorithm have been
proposed in order to increase its processing realism.[336][337] Other
researchers have argued that unsupervised forms of deep learning, such
as those based on hierarchical [generative
models](generative_model "wikilink") and [deep belief
networks](deep_belief_network "wikilink"), may be closer to biological
reality.[338][339] In this respect, generative neural network models
have been related to neurobiological evidence about sampling-based
processing in the cerebral cortex.[340]

Although a systematic comparison between the human brain organization
and the neuronal encoding in deep networks has not yet been established,
several analogies have been reported. For example, the computations
performed by deep learning units could be similar to those of actual
neurons[341] and neural populations.[342] Similarly, the representations
developed by deep learning models are similar to those measured in the
primate visual system[343] both at the single-unit[344] and at the
population[345] levels.

## Commercial activity

[Facebook](Facebook "wikilink")'s AI lab performs tasks such as
[automatically tagging uploaded
pictures](Automatic_image_annotation "wikilink") with the names of the
people in them.[346]

Google's [DeepMind Technologies](DeepMind_Technologies "wikilink")
developed a system capable of learning how to play
[Atari](Atari "wikilink") video games using only pixels as data input.
In 2015 they demonstrated their [AlphaGo](AlphaGo "wikilink") system,
which learned the game of [Go](Go_(game) "wikilink") well enough to beat
a professional Go player.[347][348][349] [Google
Translate](Google_Translate "wikilink") uses a neural network to
translate between more than 100 languages.

In 2017, Covariant.ai was launched, which focuses on integrating deep
learning into factories.[350]

As of 2008,[351] researchers at [The University of Texas at
Austin](University_of_Texas_at_Austin "wikilink") (UT) developed a
machine learning framework called Training an Agent Manually via
Evaluative Reinforcement, or TAMER, which proposed new methods for
robots or computer programs to learn how to perform tasks by interacting
with a human instructor.[352] First developed as TAMER, a new algorithm
called Deep TAMER was later introduced in 2018 during a collaboration
between [U.S. Army Research
Laboratory](U.S._Army_Research_Laboratory "wikilink") (ARL) and UT
researchers. Deep TAMER used deep learning to provide a robot with the
ability to learn new tasks through observation.[353] Using Deep TAMER, a
robot learned a task with a human trainer, watching video streams or
observing a human perform a task in-person. The robot later practiced
the task with the help of some coaching from the trainer, who provided
feedback such as "good job" and "bad job".[354]

## Criticism and comment

Deep learning has attracted both criticism and comment, in some cases
from outside the field of computer science.

### Theory

A main criticism concerns the lack of theory surrounding some
methods.[355] Learning in the most common deep architectures is
implemented using well-understood gradient descent. However, the theory
surrounding other algorithms, such as contrastive divergence is less
clear. (e.g., Does it converge? If so, how fast? What is it
approximating?) Deep learning methods are often looked at as a [black
box](black_box "wikilink"), with most confirmations done empirically,
rather than theoretically.[356]

Others point out that deep learning should be looked at as a step
towards realizing [strong
AI](Strong_artificial_intelligence "wikilink"), not as an
all-encompassing solution. Despite the power of deep learning methods,
they still lack much of the functionality needed to realize this goal
entirely. Research psychologist [Gary Marcus](Gary_Marcus "wikilink")
noted:

> Realistically, deep learning is only part of the larger challenge of
> building intelligent machines. Such techniques lack ways of
> representing [causal relationships](causality "wikilink") (...) have
> no obvious ways of performing [logical
> inferences](inference "wikilink"), and they are also still a long way
> from integrating abstract knowledge, such as information about what
> objects are, what they are for, and how they are typically used. The
> most powerful A.I. systems, like
> [Watson](Watson_(computer) "wikilink") (...) use techniques like deep
> learning as just one element in a very complicated ensemble of
> techniques, ranging from the statistical technique of [Bayesian
> inference](Bayesian_inference "wikilink") to [deductive
> reasoning](deductive_reasoning "wikilink").[357]

In further reference to the idea that artistic sensitivity might be
inherent in relatively low levels of the cognitive hierarchy, a
published series of graphic representations of the internal states of
deep (20-30 layers) neural networks attempting to discern within
essentially random data the images on which they were trained[358]
demonstrate a visual appeal: the original research notice received well
over 1,000 comments, and was the subject of what was for a time the most
frequently accessed article on *[The
Guardian](The_Guardian "wikilink")'s*[359] website.

### Errors

Some deep learning architectures display problematic behaviors,[360]
such as confidently classifying unrecognizable images as belonging to a
familiar category of ordinary images (2014)[361] and misclassifying
minuscule perturbations of correctly classified images (2013).[362]
[Goertzel](Ben_Goertzel "wikilink") hypothesized that these behaviors
are due to limitations in their internal representations and that these
limitations would inhibit integration into heterogeneous multi-component
[artificial general
intelligence](artificial_general_intelligence "wikilink") (AGI)
architectures.[363] These issues may possibly be addressed by deep
learning architectures that internally form states homologous to
image-grammar[364] decompositions of observed entities and events.[365]
[Learning a grammar](Grammar_induction "wikilink") (visual or
linguistic) from training data would be equivalent to restricting the
system to [commonsense reasoning](commonsense_reasoning "wikilink") that
operates on concepts in terms of grammatical [production
rules](Production_(computer_science) "wikilink") and is a basic goal of
both human language acquisition[366] and [artificial
intelligence](artificial_intelligence "wikilink") (AI).[367]

### Cyber threat

As deep learning moves from the lab into the world, research and
experience show that artificial neural networks are vulnerable to hacks
and deception.[368] By identifying patterns that these systems use to
function, attackers can modify inputs to ANNs in such a way that the ANN
finds a match that human observers would not recognize. For example, an
attacker can make subtle changes to an image such that the ANN finds a
match even though the image looks to a human nothing like the search
target. Such manipulation is termed an "adversarial attack".[369]

In 2016 researchers used one ANN to doctor images in trial and error
fashion, identify another's focal points, and thereby generate images
that deceived it. The modified images looked no different to human eyes.
Another group showed that printouts of doctored images then photographed
successfully tricked an image classification system.[370] One defense is
reverse image search, in which a possible fake image is submitted to a
site such as [TinEye](TinEye "wikilink") that can then find other
instances of it. A refinement is to search using only parts of the
image, to identify images from which that piece may have been
taken**.**[371]

Another group showed that certain
[psychedelic](Psychedelic_art "wikilink") spectacles could fool a
[facial recognition system](facial_recognition_system "wikilink") into
thinking ordinary people were celebrities, potentially allowing one
person to impersonate another. In 2017 researchers added stickers to
[stop signs](stop_sign "wikilink") and caused an ANN to misclassify
them.[372]

ANNs can however be further trained to detect attempts at
[deception](deception "wikilink"), potentially leading attackers and
defenders into an arms race similar to the kind that already defines the
[malware](malware "wikilink") defense industry. ANNs have been trained
to defeat ANN-based anti-[malware](malware "wikilink") software by
repeatedly attacking a defense with malware that was continually altered
by a [genetic algorithm](genetic_algorithm "wikilink") until it tricked
the anti-malware while retaining its ability to damage the target.[373]

In 2016, another group demonstrated that certain sounds could make the
[Google Now](Google_Now "wikilink") voice command system open a
particular web address, and hypothesized that this could "serve as a
stepping stone for further attacks (e.g., opening a web page hosting
drive-by malware)".[374]

In "[data
poisoning](Adversarial_machine_learning#Data_poisoning "wikilink")",
false data is continually smuggled into a machine learning system's
training set to prevent it from achieving mastery.[375]

### Data collection ethics

Most Deep Learning systems rely on training and verification data that
is generated and/or annotated by humans.[376] It has been argued in
[media philosophy](Media_studies "wikilink") that not only low-paid
[clickwork](Clickworkers "wikilink") (e.g. on [Amazon Mechanical
Turk](Amazon_Mechanical_Turk "wikilink")) is regularly deployed for this
purpose, but also implicit forms of human
[microwork](microwork "wikilink") that are often not recognized as
such.[377] The philosopher [Rainer Mühlhoff](Rainer_Mühlhoff "wikilink")
distinguishes five types of "machinic capture" of human microwork to
generate training data: (1) [gamification](gamification "wikilink") (the
embedding of annotation or computation tasks in the flow of a game), (2)
"trapping and tracking" (e.g. [CAPTCHAs](CAPTCHA "wikilink") for image
recognition or click-tracking on Google [search results
pages](Search_engine_results_page "wikilink")), (3) exploitation of
social motivations (e.g. [tagging faces](Tag_(Facebook) "wikilink") on
[Facebook](Facebook "wikilink") to obtain labeled facial images), (4)
[information mining](information_mining "wikilink") (e.g. by leveraging
[quantified-self](Quantified_self "wikilink") devices such as [activity
trackers](activity_tracker "wikilink")) and (5)
[clickwork](Clickworkers "wikilink").[378]

Mühlhoff argues that in most commercial end-user applications of Deep
Learning such as [Facebook's face recognition
system](DeepFace "wikilink"), the need for training data does not stop
once an ANN is trained. Rather, there is a continued demand for
human-generated verification data to constantly calibrate and update the
ANN. For this purpose, Facebook introduced the feature that once a user
is automatically recognized in an image, they receive a notification.
They can choose whether or not they like to be publicly labeled on the
image, or tell Facebook that it is not them in the picture.[379] This
user interface is a mechanism to generate "a constant stream of
verification data"[380] to further train the network in real-time. As
Mühlhoff argues, the involvement of human users to generate training and
verification data is so typical for most commercial end-user
applications of Deep Learning that such systems may be referred to as
"human-aided artificial intelligence".[381]

## See also

-   [Applications of artificial
    intelligence](Applications_of_artificial_intelligence "wikilink")
-   [Comparison of deep learning
    software](Comparison_of_deep_learning_software "wikilink")
-   [Compressed sensing](Compressed_sensing "wikilink")
-   [Differentiable programming](Differentiable_programming "wikilink")
-   [Echo state network](Echo_state_network "wikilink")
-   [List of artificial intelligence
    projects](List_of_artificial_intelligence_projects "wikilink")
-   [Liquid state machine](Liquid_state_machine "wikilink")
-   [List of datasets for machine-learning
    research](List_of_datasets_for_machine-learning_research "wikilink")
-   [Reservoir computing](Reservoir_computing "wikilink")
-   [Scale space and deep
    learning](Scale_space#Deep_learning_and_scale_space "wikilink")
-   [Sparse coding](Sparse_coding "wikilink")
-   [Stochastic parrot](Stochastic_parrot "wikilink")
-   [Topological deep learning](Topological_deep_learning "wikilink")

## References

## Further reading

-   

-   

-   

[ ](Category:Deep_learning "wikilink") [Category:Artificial neural
networks](Category:Artificial_neural_networks "wikilink")

[1]

[2]

[3]

[4]

[5]

[6]
Massachusetts Institute of Technology |language=en}}

[7]

[8]

[9]

[10]

[11]

[12]

[13]

[14]

[15]

[16]

[17]

[18]

[19]

[20] Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). [The
Expressive Power of Neural Networks: A View from the
Width](http://papers.nips.cc/paper/7203-the-expressive-power-of-neural-networks-a-view-from-the-width)
. Neural Information Processing Systems, 6231-6239.

[21]

[22]

[23]

[24]

[25]

[26]

[27]

[28]

[29]

[30]

[31]

[32]

[33]

[34]

[35]

[36]

[37]

[38]

[39]

[40]

[41]

[42]

[43]

[44]

[45]

[46]

[47]

[48]

[49]

[50]

[51]

[52]

[53]

[54]

[55]

[56]

[57]

[58]

[59] Matthew Brand (1988) Machine and Brain Learning. University of
Chicago Tutorial Studies Bachelor's Thesis, 1988. Reported at the Summer
Linguistics Institute, Stanford University, 1987

[60]

[61]

[62]

[63]

[64]

[65]

[66]

[67]

[68]

[69]

[70]

[71]

[72]

[73] Rumelhart, David E., Geoffrey E. Hinton, and R. J. Williams.
"[Learning Internal Representations by Error
Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)".
David E. Rumelhart, James L. McClelland, and the PDP research group.
(editors), Parallel distributed processing: Explorations in the
microstructure of cognition, Volume 1: Foundation. MIT Press, 1986.

[74]

[75]

[76]

[77]

[78] [Rina Dechter](Rina_Dechter "wikilink") (1986). Learning while
searching in constraint-satisfaction problems. University of California,
Computer Science Department, Cognitive Systems
Laboratory.[Online](https://www.researchgate.net/publication/221605378_Learning_While_Searching_in_Constraint-Satisfaction-Problems)

[79]

[80] Co-evolving recurrent neurons learn deep memory POMDPs. Proc.
GECCO, Washington, D. C., pp. 1795–1802, ACM Press, New York, NY, USA,
2005.

[81]

[82]

[83] LeCun *et al.*, "Backpropagation Applied to Handwritten Zip Code
Recognition", *Neural Computation*, 1, pp. 541–551, 1989.

[84]

[85]

[86]

[87]

[88]

[89]

[90]

[91]

[92]

[93]

[94]

[95]

[96]

[97]

[98]

[99]

[100]

[101]

[102]

[103]

[104]

[105]

[106]

[107]

[108] S. Hochreiter., "[Untersuchungen zu dynamischen neuronalen
Netzen](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)".
. *Diploma thesis. Institut f. Informatik, Technische Univ. Munich.
Advisor: J. Schmidhuber*, 1991.

[109]

[110]

[111]

[112]

[113]

[114]

[115]

[116]

[117]

[118]

[119]

[120]

[121]

[122]

[123]

[124]

[125]

[126]

[127]

[128]

[129]

[130]

[131]

[132]

[133]

[134]

[135]

[136]

[137] Santiago Fernandez, Alex Graves, and Jürgen Schmidhuber (2007).
[An application of recurrent neural networks to discriminative keyword
spotting](https://mediatum.ub.tum.de/doc/1289941/file.pdf) . Proceedings
of ICANN (2), pp. 220–229.

[138]

[139] [Yann LeCun](Yann_LeCun "wikilink") (2016). Slides on Deep
Learning [Online](https://indico.cern.ch/event/510372/)

[140]

[141]

[142]

[143] G. E. Hinton., "[Learning multiple layers of
representation](http://www.csri.utoronto.ca/~hinton/absps/ticsdraft.pdf)".
. *Trends in Cognitive Sciences*, 11, pp. 428–434, 2007.

[144]

[145]

[146]

[147]

[148]

[149]

[150]

[151]

[152]

[153]

[154]

[155]

[156]

[157]

[158]

[159]

[160]

[161]

[162]

[163]

[164]

[165]

[166]

[167]

[168] "[A Survey of Techniques for Optimizing Deep Learning on
GPUs](https://www.academia.edu/40135801) ", S. Mittal and S. Vaishay,
Journal of Systems Architecture, 2019

[169]

[170]

[171]

[172]

[173]

[174] Graves, Alex; and Schmidhuber, Jürgen; *Offline Handwriting
Recognition with Multidimensional Recurrent Neural Networks*, in Bengio,
Yoshua; Schuurmans, Dale; Lafferty, John; Williams, Chris K. I.; and
Culotta, Aron (eds.), *Advances in Neural Information Processing Systems
22 (NIPS'22), December 7th–10th, 2009, Vancouver, BC*, Neural
Information Processing Systems (NIPS) Foundation, 2009, pp. 545–552

[175]

[176] Google Research Blog. The neural networks behind Google Voice
transcription. August 11, 2015. By Françoise Beaufays
<http://googleresearch.blogspot.co.at/2015/08/the-neural-networks-behind-google-voice.html>

[177]

[178]

[179]

[180]

[181]

[182]

[183]

[184]

[185]

[186]

[187]

[188]

[189]

[190]

[191]
.

[192]
.

[193]
.

[194]

[195]
Data Science Association|website=www.datascienceassn.org|access-date=14
June 2017|archive-date=30 April
2017|archive-url=<https://web.archive.org/web/20170430142049/http://www.datascienceassn.org/content/multi-task-neural-networks-qsar-predictions%7Curl-status=live>}}

[196] "Toxicology in the 21st century Data Challenge"

[197]

[198]

[199]

[200]

[201]

[202]

[203]

[204]

[205]

[206]

[207]

[208]

[209]

[210]

[211]

[212]

[213]

[214]

[215]

[216]

[217]

[218]

[219]

[220]
Coursera|website=Coursera|access-date=30 November 2017|archive-date=1
December
2017|archive-url=<https://web.archive.org/web/20171201032606/https://www.coursera.org/learn/convolutional-neural-networks/lecture/AYzbX/data-augmentation%7Curl-status=live>}}

[221]

[222]

[223]

[224] Ting Qin, et al. "A learning algorithm of CMAC based on RLS".
Neural Processing Letters 19.1 (2004): 49-61.

[225] Ting Qin, et al. "[Continuous CMAC-QRLS and its systolic
array](http://www-control.eng.cam.ac.uk/Homepage/papers/cued_control_997.pdf)".
. Neural Processing Letters 22.1 (2005): 1-16.

[226]

[227]

[228]

[229]

[230]

[231]

[232]

[233]

[234]

[235]

[236]

[237]

[238]

[239]

[240]

[241]

[242]

[243]

[244]

[245]

[246]
WIRED|magazine=Wired|access-date=14 June 2017|date=17 December
2014|last1=McMillan|first1=Robert|archive-date=8 June
2017|archive-url=<https://web.archive.org/web/20170608062106/https://www.wired.com/2014/12/skype-used-ai-build-amazing-new-language-translator/%7Curl-status=live>}}

[247]

[248]

[249]

[250]

[251] [Nvidia Demos a Car Computer Trained with "Deep
Learning"](http://www.technologyreview.com/news/533936/nvidia-demos-a-car-computer-trained-with-deep-learning/)
(6 January 2015), David Talbot, *[MIT Technology
Review](MIT_Technology_Review "wikilink")*

[252]

[253]

[254]

[255]

[256]

[257]

[258]

[259]

[260]

[261]

[262]

[263]

[264]

[265]

[266]

[267]

[268]

[269]

[270]

[271]

[272]

[273]

[274]

[275]

[276]

[277]

[278]

[279]

[280]

[281]

[282]

[283]

[284]

[285]

[286]

[287]

[288]

[289]

[290]

[291]

[292]

[293]

[294]

[295]

[296]

[297]

[298]

[299]

[300]

[301]

[302]

[303]

[304]

[305]

[306]

[307]

[308]

[309]

[310]

[311]

[312]

[313]

[314]

[315]

[316]

[317]

[318]

[319]

[320]

[321]

[322]

[323]

[324]

[325]

[326]

[327]

[328]

[329]

[330]

[331]

[332]

[333]

[334]

[335] S. Blakeslee, "In brain's early growth, timetable may be
critical", *The New York Times, Science Section*, pp. B5–B6, 1995.

[336]

[337]

[338]

[339]

[340]

[341]

[342]

[343]

[344]

[345]

[346]

[347]

[348]

[349]
MIT Technology Review|url =
<http://www.technologyreview.com/news/546066/googles-ai-masters-the-game-of-go-a-decade-earlier-than-expected/%7Cwebsite>
= MIT Technology Review|access-date = 30 January 2016|archive-date = 1
February 2016|archive-url =
<https://web.archive.org/web/20160201140636/http://www.technologyreview.com/news/546066/googles-ai-masters-the-game-of-go-a-decade-earlier-than-expected/%7Curl-status>
= dead}}

[350]

[351]

[352]

[353]

[354]

[355]

[356]

[357]

[358]

[359]

[360]

[361]

[362]

[363]

[364]

[365]

[366] Miller, G. A., and N. Chomsky. "Pattern conception". Paper for
Conference on pattern detection, University of Michigan. 1957.

[367]

[368]

[369]

[370]

[371]

[372]

[373]

[374]

[375]

[376]

[377]

[378]

[379]

[380]

[381]
