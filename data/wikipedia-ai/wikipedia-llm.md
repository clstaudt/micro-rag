A **large language model** (**LLM**) is a computational
[model](Model#Conceptual_model "wikilink") notable for its ability to
achieve general-purpose language generation and other [natural language
processing](natural_language_processing "wikilink") tasks such as
[classification](Statistical_classification "wikilink"). Based on
[language models](language_model "wikilink"), LLMs acquire these
abilities by learning statistical relationships from vast amounts of
text during a computationally intensive
[self-supervised](self-supervised_learning "wikilink") and
[semi-supervised](semi-supervised_learning "wikilink") training
process.[1] LLMs can be used for text generation, a form of [generative
AI](Generative_artificial_intelligence "wikilink"), by taking an input
text and repeatedly predicting the next token or word.[2]

LLMs are [artificial neural
networks](artificial_neural_network "wikilink") that utilize the
[transformer](Transformer_(deep_learning_architecture) "wikilink")
architecture, invented in 2017. The largest and most capable LLMs, , are
built with a decoder-only transformer-based architecture, which enables
efficient processing and generation of large-scale text data.

Historically, up to 2020,
[fine-tuning](Fine-tuning_(deep_learning) "wikilink") was the primary
method used to adapt a model for specific tasks. However, larger models
such as [GPT-3](GPT-3 "wikilink") have demonstrated the ability to
achieve similar results through [prompt
engineering](prompt_engineering "wikilink"), which involves crafting
specific input prompts to guide the model's responses.[3] These models
acquire knowledge about syntax, semantics, and
[ontologies](ontology_(information_science) "wikilink")[4] inherent in
human language corpora, but they also inherit inaccuracies and
[biases](Algorithmic_bias "wikilink") present in the data they are
trained on.[5]

Some notable LLMs are [OpenAI](OpenAI "wikilink")'s
[GPT](Generative_pre-trained_transformer "wikilink") series of models
(e.g., [GPT-3.5](GPT-3.5 "wikilink") and [GPT-4](GPT-4 "wikilink"), used
in [ChatGPT](ChatGPT "wikilink") and [Microsoft
Copilot](Microsoft_Copilot "wikilink")), [Google](Google "wikilink")'s
[Gemini](Gemini_(language_model) "wikilink") (the latter of which is
currently used in [the chatbot of the same
name](Gemini_(chatbot) "wikilink")), [Meta](Meta_Platforms "wikilink")'s
[LLaMA](LLaMA "wikilink") family of models,
[Anthropic](Anthropic "wikilink")'s
[Claude](Claude_(language_model) "wikilink") models, and [Mistral
AI](Mistral_AI "wikilink")'s models.

## History

[thumb|upright=1.3|An illustration of main components of the transformer
model from the original paper, where layers were normalized after
(instead of before) multiheaded
attention](File:The-Transformer-model-architecture.png "wikilink") At
the 2017 [NeurIPS](NeurIPS "wikilink") conference, Google researchers
introduced the [transformer
architecture](transformer_architecture "wikilink") in their landmark
paper "[Attention Is All You
Need](Attention_Is_All_You_Need "wikilink")". This paper's goal was to
improve upon 2014 [Seq2seq](Seq2seq "wikilink") technology, [6] and was
based mainly on the [attention](attention_(machine_learning) "wikilink")
mechanism developed by Bahdanau et al. in 2014.[7] The following year in
2018, [BERT](BERT_(language_model) "wikilink") was introduced and
quickly became "ubiquitous".[8] Though the original transformer has both
encoder and decoder blocks, BERT is an encoder-only model.

Although decoder-only [GPT-1](GPT-1 "wikilink") was introduced in 2018,
it was [GPT-2](GPT-2 "wikilink") in 2019 that caught widespread
attention because [OpenAI](OpenAI "wikilink") at first deemed it too
powerful to release publicly, out of fear of malicious use.[9]
[GPT-3](GPT-3 "wikilink") in 2020 went a step further and is available
only via [API](Web_API "wikilink") with no offering of downloading the
model to execute locally. But it was the 2022 consumer-facing
browser-based [ChatGPT](ChatGPT "wikilink") that captured the
imaginations of the general population and caused some media hype and
online buzz.[10] The 2023 [GPT-4](GPT-4 "wikilink") was praised for its
increased accuracy and as a "holy grail" for its
[multimodal](Multimodal_learning "wikilink") capabilities.[11] OpenAI
did not reveal high-level architecture and the number of
[parameters](Parameter#Artificial_Intelligence "wikilink") of GPT-4.

Competing language models have for the most part been attempting to
equal the GPT series, at least in terms of number of parameters.[12]

Since 2022, [source-available](Source-available_software "wikilink")
models have been gaining popularity, especially at first with
[BLOOM](BLOOM_(language_model) "wikilink") and
[LLaMA](LLaMA "wikilink"), though both have restrictions on the field of
use. [Mistral AI](Mistral_AI "wikilink")'s models Mistral 7B and Mixtral
8x7b have the more permissive [Apache
License](Apache_License "wikilink"). , Mixtral 8x7b is the most powerful
open LLM according to the LMSYS Chatbot Arena Leaderboard, being more
powerful than GPT-3.5 but not as powerful as GPT-4.[13]

### Alternative architecture

As of 2024, the largest and most capable models are all based on the
Transformer architecture. Some recent implementations are based on other
architectures, such as [recurrent neural
network](recurrent_neural_network "wikilink") variants and
[Mamba](Mamba_(deep_learning_architecture) "wikilink") (a [state
space](state-space_representation "wikilink") model).[14][15][16]

## Dataset preprocessing

### Probabilistic tokenization

Because [machine learning](machine_learning "wikilink") algorithms
process numbers rather than text, the text must be converted to numbers.
In the first step, a vocabulary is decided upon, then integer indexes
are arbitrarily but uniquely assigned to each vocabulary entry, and
finally, an [embedding](Word_embedding "wikilink") is associated to the
integer index. Algorithms include [byte-pair
encoding](byte_pair_encoding "wikilink") and
[WordPiece](BERT_(language_model)#Design "wikilink").

Probabilistic tokenization also
[compresses](Data_compression "wikilink") the datasets. Because LLMs
generally require input to be an
[array](Array_(data_structure) "wikilink") that is not
[jagged](Jagged_array "wikilink"), the shorter texts must be "padded"
until they match the length of the longest one. How many tokens are, on
average, needed per word depends on the language of the dataset.[17][18]

#### BPE

Using a modification of byte-pair encoding, in the first step, all
unique characters (including blanks and [punctuation
marks](punctuation_mark "wikilink")) are treated as an initial set of
[*n*-grams](n-gram "wikilink") (i.e. initial set of uni-grams).
Successively the most frequent pair of adjacent characters is merged
into a bi-gram and all instances of the pair are replaced by it. All
occurrences of adjacent pairs of (previously merged) *n*-grams that most
frequently occur together are then again merged into even lengthier
*n*-gram repeatedly until a vocabulary of prescribed size is obtained
(in case of [GPT-3](GPT-3 "wikilink"), the size is 50257).[19] Token
vocabulary consists of [integers](integers "wikilink"), spanning from
zero up to the size of the token vocabulary. New words can always be
interpreted as combinations of the tokens and the initial-set
uni-grams.[20]

A token vocabulary based on the frequencies extracted from mainly
English corpora uses as few tokens as possible for an average English
word. An average word in another language encoded by such an
English-optimized tokenizer is however split into suboptimal amount of
tokens. GPT-2 tokenizer can use up to 15 times more tokens per word for
some languages, for example for the [Shan
language](Shan_language "wikilink") from [Myanmar](Myanmar "wikilink").
Even more widespread languages such as Portuguese and German have "a
premium of 50%" compared to English.[21]

For example, here is how tokenizer used by GPT-3 (Legacy) split the
following sentence
<small>`tokenizer: texts -> series of numerical "tokens"`</small>.

<table>
<tbody>
<tr class="odd">
<td
style="border-left: 2px green; border-right: 2px green"><p>token</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p>izer</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p>:</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p> texts</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p> -&gt;</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p>series</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p> of</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p> numerical</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p> "</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p>t</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p>ok</p></td>
<td
style="background-color: grey; color: white; border-left: 2px green; border-right: 2px green"><p>ens</p></td>
<td
style="border-left: 2px green; border-right: 2px green"><p>"</p></td>
</tr>
</tbody>
</table>

### Dataset cleaning

In the context of training LLMs, datasets are typically cleaned by
removing toxic passages from the dataset, discarding low-quality data,
and de-duplication.[22] Cleaned datasets can increase training
efficiency and lead to improved downstream performance.[23][24] A
trained LLM can be used to clean datasets for training a further
LLM.[25]

With the increasing proportion of LLM-generated content on the web, data
cleaning in the future may include filtering out such content.
LLM-generated content can pose a problem if the content is similar to
human text (making filtering difficult) but of lower quality (degrading
performance of models trained on it).[26]

### Synthetic data

Training of largest language models might need more linguistic data than
naturally available, or that the naturally occurring data is of
insufficient quality. In these cases, synthetic data might be used. The
Microsoft's Phi series of LLMs is trained on textbook-like data
generated by another LLM.[27]

## Training and architecture

### Reinforcement learning from human feedback (RLHF)

[Reinforcement learning from human
feedback](Reinforcement_learning_from_human_feedback "wikilink") (RLHF)
through algorithms, such as [proximal policy
optimization](Proximal_Policy_Optimization "wikilink"), is used to
further fine-tune a model based on a dataset of human preferences.[28]

### Instruction tuning

Using "self-instruct" approaches, LLMs have been able to
[bootstrap](Bootstrapping "wikilink") correct responses, replacing any
naive responses, starting from human-generated corrections of a few
cases. For example, in the instruction "Write an essay about the main
themes represented in Hamlet," an initial naive completion might be "If
you submit the essay after March 17, your grade will be reduced by 10%
for each day of delay," based on the frequency of this textual sequence
in the corpus.[29]

### Mixture of experts

The largest LLM may be too expensive to train and use directly. For such
models, [mixture of experts](mixture_of_experts "wikilink") (MoE) can be
applied, a line of research pursued by Google researchers since 2017 to
train models reaching up to 1 trillion parameters.[30][31][32]

### Prompt engineering, attention mechanism, and context window

Most results previously achievable only by (costly) fine-tuning, can be
achieved through [prompt engineering](prompt_engineering "wikilink"),
although limited to the scope of a single conversation (more precisely,
limited to the scope of a context window).[33] [300px|thumb | When each
head calculates, according to its own criteria, how much other tokens
are relevant for the "it\_" token, note that the second attention head,
represented by the second column, is focusing most on the first two
rows, i.e. the tokens "The" and "animal", while the third column is
focusing most on the bottom two rows, i.e. on "tired", which has been
tokenized into two
tokens.[34]](File:Multiple_attention_heads.png "wikilink")

In order to find out which tokens are relevant to each other within the
scope of the context window, the attention mechanism calculates "soft"
weights for each token, more precisely for its embedding, by using
multiple attention heads, each with its own "relevance" for calculating
its own soft weights. For example, the small (i.e. 117M parameter sized)
[GPT-2](GPT-2 "wikilink") model, has had twelve attention heads and a
context window of only 1k token.[35] In its medium version it has 345M
parameters and contains 24 layers, each with 12 attention heads. For the
training with gradient descent a batch size of 512 was utilized.[36]

The largest models, such as Google's [Gemini
1.5](Gemini_(language_model) "wikilink"), presented in February 2024,
can have a context window sized up to 1 million (context window of 10
million was also "successfully tested").[37] Other models with large
context windows includes Anthropic's Claude 2.1, with a context window
of up to 200k tokens.[38] Note that this maximum refers to the number of
input tokens and that the maximum number of output tokens differs from
the input and is often smaller. For example, the GPT-4 Turbo model has a
maximum output of 4096 tokens.[39]

Length of a conversation that the model can take into account when
generating its next answer is limited by the size of a context window,
as well. If the length of a conversation, for example with
[Chat-GPT](Chat-GPT "wikilink"), is longer than its context window, only
the parts inside the context window are taken into account when
generating the next answer, or the model needs to apply some algorithm
to summarize the too distant parts of conversation.

The shortcomings of making a context window larger include higher
computational cost and possibly diluting the focus on local context,
while making it smaller can cause a model to miss an important
long-range dependency. Balancing them are a matter of experimentation
and domain-specific considerations.

A model may be pre-trained either to predict how the segment continues,
or what is missing in the segment, given a segment from its training
dataset.[40] It can be either

-   autoregressive (i.e. predicting how the segment continues, the way
    [GPTs](Generative_pretrained_transformer "wikilink") do it): for
    example given a segment "I like to eat", the model predicts "ice
    cream", or "sushi".
-   "[masked](Cloze_test "wikilink")" (i.e. filling in the parts missing
    from the segment, the way "BERT"[41] does it): for example, given a
    segment "I like to `[__] [__]` cream", the model predicts that "eat"
    and "ice" are missing.

Models may be trained on auxiliary tasks which test their understanding
of the data distribution, such as Next Sentence Prediction (NSP), in
which pairs of sentences are presented and the model must predict
whether they appear consecutively in the training corpus.[42] During
training, [regularization](Regularization_(mathematics) "wikilink") loss
is also used to stabilize training. However regularization loss is
usually not used during
[testing](Training,_validation,_and_test_data_sets "wikilink") and
evaluation.

## Training cost

Advances in software and hardware have reduced the cost substantially
since 2020, such that in 2023 training of a 12-billion-parameter LLM
computational cost is 72,300
[A100-GPU](Ampere_(microarchitecture) "wikilink")-hours, while in 2020
the cost of training a 1.5-billion-parameter LLM (which was two orders
of magnitude smaller than the state of the art in 2020) was between $80
thousand and $1.6 million.[43][44][45] Since 2020, large sums were
invested in increasingly large models. For example, training of the
GPT-2 (i.e. a 1.5-billion-parameters model) in 2019 cost $50,000, while
training of the PaLM (i.e. a 540-billion-parameters model) in 2022 cost
$8 million, and Megatron-Turing NLG 530B (in 2021) cost around $11
million.[46]

For Transformer-based LLM, training cost is much higher than inference
cost. It costs 6 [FLOPs](FLOPS "wikilink") per parameter to train on one
token, whereas it costs 1 to 2 FLOPs per parameter to infer on one
token.[47]

## Tool use

There are certain tasks that, in principle, cannot be solved by any LLM,
at least not without the use of external tools or additional software.
An example of such a task is responding to the user's input '354 \* 139
= ', provided that the LLM has not already encountered a continuation of
this calculation in its training corpus. In such cases, the LLM needs to
resort to running program code that calculates the result, which can
then be included in its response. Another example is 'What is the time
now? It is ', where a separate program interpreter would need to execute
a code to get system time on the computer, so LLM could include it in
its reply.[48][49] This basic strategy can be sophisticated with
multiple attempts of generated programs, and other sampling
strategies.[50]

Generally, in order to get an LLM to use tools, one must finetune it for
tool-use. If the number of tools is finite, then finetuning may be done
just once. If the number of tools can grow arbitrarily, as with online
[API](API "wikilink") services, then the LLM can be fine-tuned to be
able to read API documentation and call API correctly.[51][52]

A simpler form of tool use is *Retrieval Augmented Generation*: augment
an LLM with [document retrieval](document_retrieval "wikilink"),
sometimes using a [vector database](vector_database "wikilink"). Given a
query, a document retriever is called to retrieve the most relevant
(usually measured by first encoding the query and the documents into
vectors, then finding the documents with vectors closest in Euclidean
norm to the query vector). The LLM then generates an output based on
both the query and the retrieved documents.[53]

## Agency

An LLM is a language model, which is not an agent as it has no goal, but
it can be used as a component of an [intelligent
agent](intelligent_agent "wikilink").[54] Researchers have described
several methods for such integrations.

The ReAct ("Reason + Act") method constructs an
[agent](Intelligent_agent "wikilink") out of an LLM, using the LLM as a
planner. The LLM is prompted to "think out loud". Specifically, the
language model is prompted with a textual description of the
environment, a goal, a list of possible actions, and a record of the
actions and observations so far. It generates one or more thoughts
before generating an action, which is then executed in the
environment.[55] The linguistic description of the environment given to
the LLM planner can even be the LaTeX code of a paper describing the
environment.[56]

In the DEPS ("Describe, Explain, Plan and Select") method, an LLM is
first connected to the visual world via image descriptions, then it is
prompted to produce plans for complex tasks and behaviors based on its
pretrained knowledge and environmental feedback it receives.[57]

The Reflexion method[58] constructs an agent that learns over multiple
episodes. At the end of each episode, the LLM is given the record of the
episode, and prompted to think up "lessons learned", which would help it
perform better at a subsequent episode. These "lessons learned" are
given to the agent in the subsequent episodes.

[Monte Carlo tree search](Monte_Carlo_tree_search "wikilink") can use an
LLM as rollout heuristic. When a programmatic world model is not
available, an LLM can also be prompted with a description of the
environment to act as world model.[59]

For open-ended exploration, an LLM can be used to score observations for
their "interestingness", which can be used as a reward signal to guide a
normal (non-LLM) reinforcement learning agent.[60] Alternatively, it can
[propose increasingly difficult
tasks](Zone_of_proximal_development "wikilink") for [curriculum
learning](curriculum_learning "wikilink").[61] Instead of outputting
individual actions, an LLM planner can also construct "skills", or
[functions](Function_(computer_programming) "wikilink") for complex
action sequences. The skills can be stored and later invoked, allowing
increasing levels of abstraction in planning.[62]

LLM-powered agents can keep a long-term memory of its previous contexts,
and the memory can be retrieved in the same way as Retrieval Augmented
Generation. Multiple such agents can interact socially.[63]

## Compression

Typically, LLM are trained with single- or half-precision floating point
numbers (float32 and float16). One float16 has 16 bits, or 2 bytes, and
so one billion parameters require 2 gigabytes. The largest models
typically have 100 billion parameters, requiring 200 gigabytes to load,
which places them outside the range of most consumer electronics.[64]

*Post-training
[quantization](Quantization_(signal_processing) "wikilink")*[65] aims to
decrease the space requirement by lowering precision of the parameters
of a trained model, while preserving most of its performance.[66][67]
The simplest form of quantization simply truncates all numbers to a
given number of bits. It can be improved by using a different
quantization [codebook](Block_cipher "wikilink") per layer. Further
improvement can be done by applying [different
precisions](Mixed-precision_arithmetic "wikilink") to different
parameters, with higher precision for particularly important parameters
("outlier weights").[68]

While quantized models are typically frozen, and only pre-quantized
models are fine-tuned, quantized models can still be fine-tuned.[69]

## Multimodality

Multimodality means "having several modalities", and a
["modality"](Modality_(human–computer_interaction) "wikilink") refers to
a type of input or output, such as video, image, audio, text,
[proprioception](proprioception "wikilink"), etc.[70] There have been
many AI models trained specifically to ingest one modality and output
another modality, such as [AlexNet](AlexNet "wikilink") for image to
label,[71] [visual question
answering](visual_question_answering "wikilink") for image-text to
text,[72] and [speech recognition](speech_recognition "wikilink") for
speech to text.

A common method to create multimodal models out of an LLM is to
"tokenize" the output of a trained encoder. Concretely, one can
construct a LLM that can understand images as follows: take a trained
LLM, and take a trained image encoder *E*. Make a small multilayered
perceptron *f*, so that for any image *y*, the post-processed vector
*f*(*E*(*y*)) has the same dimensions as an encoded token. That is an
"image token". Then, one can interleave text tokens and image tokens.
The compound model is then fine-tuned on an image-text dataset. This
basic construction can be applied with more sophistication to improve
the model. The image encoder may be frozen to improve stability.[73]

Flamingo demonstrated the effectiveness of the tokenization method,
finetuning a pair of pretrained language model and image encoder to
perform better on visual question answering than models trained from
scratch.[74] [Google PaLM](Pathways_Language_Model "wikilink") model was
fine-tuned into a multimodal model PaLM-E using the tokenization method,
and applied to robotic control.[75] [LLaMA](LLaMA "wikilink") models
have also been turned multimodal using the tokenization method, to allow
image inputs,[76] and video inputs.[77]

[GPT-4](GPT-4 "wikilink") can use both text and image as inputs[78]
(although the vision component wasn't released to the public until
GPT-4V[79]); [Google DeepMind](Google_DeepMind "wikilink")'s
[Gemini](Gemini_(language_model) "wikilink") is also multimodal.[80]

## Properties

### Scaling laws

The following four hyper-parameters characterize a LLM:

-   cost of (pre-)training (<small>*C*</small>),
-   size of the [artificial neural
    network](artificial_neural_network "wikilink") itself, such as
    number of parameters <small>*N*</small> (i.e. amount of neurons in
    its layers, amount of weights between them and biases),
-   size of its (pre-)training dataset (i.e. number of tokens in corpus,
    <small>*D*</small>),
-   performance after (pre-)training.

They are related by simple [statistical
laws](Empirical_statistical_laws "wikilink"), called "scaling laws". One
particular scaling law ("[Chinchilla
scaling](Chinchilla_AI "wikilink")") for LLM autoregressively trained
for one epoch, with a [log-log](Log-log_plot "wikilink") [learning
rate](learning_rate "wikilink") schedule, states that:[81]
$\begin{cases}
C = C\_0 ND \\\[6pt\]
L = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L\_0
\end{cases}$ where the variables are

-   <small>*C*</small> is the cost of training the model, in
    [FLOPs](FLOPS "wikilink").
-   <small>*N*</small> is the number of parameters in the model.
-   <small>*D*</small> is the number of tokens in the training set.
-   <small>*L*</small> is the average negative log-likelihood loss per
    token ([nats](Nat_(unit) "wikilink")/token), achieved by the trained
    LLM on the test dataset.

and the statistical hyper-parameters are

-   <small>*C*<sub>0</sub> = 6</small>, meaning that it costs 6 FLOPs
    per parameter to train on one token. Note that training cost is much
    higher than inference cost, where it costs 1 to 2 FLOPs per
    parameter to infer on one token.[82]
-   <small>*α* = 0.34, *β* = 0.28, *A* = 406.4, *B* = 410.7, *L*<sub>0</sub> = 1.69</small>

### Emergent abilities

![At point(s) referred to as
[breaks](Neural_scaling_law#Broken_Neural_Scaling_Laws_(BNSL) "wikilink"),[83]
the lines change their slopes, appearing on a linear-log plot as a
series of linear segments connected by
arcs.](LLM_emergent_benchmarks.png "At point(s) referred to as breaks, the lines change their slopes, appearing on a linear-log plot as a series of linear segments connected by arcs.")
Performance of bigger models on various tasks, when plotted on a log-log
scale, appears as a linear extrapolation of performance achieved by
smaller models. However, this linearity may be punctuated by
"[break(s)](Neural_scaling_law#Broken_Neural_Scaling_Laws_(BNSL) "wikilink")"[85]
in the scaling law, where the slope of the line changes abruptly, and
where larger models acquire "emergent abilities".[86][87] They arise
from the complex interaction of the model's components and are not
explicitly programmed or designed.[88]

The most intriguing among emergent abilities is [in-context
learning](in-context_learning "wikilink") from example
demonstrations.[89] In-context learning is involved in tasks, such as:

-   reported arithmetics, decoding the [International Phonetic
    Alphabet](International_Phonetic_Alphabet "wikilink"), unscrambling
    a word's letters, disambiguate word in context,[90][91][92]
    converting spatial words, [cardinal
    directions](cardinal_direction "wikilink") (for example, replying
    "northeast" upon \[0, 0, 1; 0, 0, 0; 0, 0, 0\]), color terms
    represented in text.[93]
-   [chain-of-thought prompting](chain-of-thought_prompting "wikilink"):
    Model outputs are improved by chain-of-thought prompting only when
    model size exceeds 62B. Smaller models perform better when prompted
    to answer immediately, without chain of thought.[94]
-   identifying offensive content in paragraphs of
    [Hinglish](Hinglish "wikilink") (a combination of Hindi and
    English), and generating a similar English equivalent of
    [Kiswahili](Kiswahili "wikilink") proverbs.[95]

Schaeffer *et. al.* argue that the emergent abilities are not
unpredictably acquired, but predictably acquired according to a [smooth
scaling law](Neural_scaling_law "wikilink"). The authors considered a
toy statistical model of an LLM solving multiple-choice questions, and
showed that this statistical model, modified to account for other types
of tasks, applies to these tasks as well.[96]

Let *x* be the number of parameter count, and *y* be the performance of
the model.

## Interpretation

Large language models by themselves are "[black
boxes](black_box "wikilink")", and it is not clear how they can perform
linguistic tasks. There are several methods for understanding how LLM
work.

Mechanistic interpretability aims to
[reverse-engineer](Reverse_engineering "wikilink") LLM by discovering
symbolic algorithms that approximate the inference performed by LLM. One
example is Othello-GPT, where a small Transformer is trained to predict
legal [Othello](reversi "wikilink") moves. It is found that there is a
linear representation of Othello board, and modifying the representation
changes the predicted legal Othello moves in the correct way.[97][98] In
another example, a small Transformer is trained on [Karel
programs](Karel_(programming_language) "wikilink"). Similar to the
Othello-GPT example, there is a linear representation of Karel program
semantics, and modifying the representation changes output in the
correct way. The model also generates correct programs that are on
average shorter than those in the training set.[99]

In another example, the authors trained small transformers on [modular
arithmetic addition](Modular_arithmetic "wikilink"). The resulting
models were reverse-engineered, and it turned out they used [discrete
Fourier transform](discrete_Fourier_transform "wikilink").[100]

### Understanding and intelligence

NLP researchers were evenly split when asked, in a 2022 survey, whether
(untuned) LLMs "could (ever) understand natural language in some
nontrivial sense".[101] Proponents of "LLM understanding" believe that
some LLM abilities, such as mathematical reasoning, imply an ability to
["understand"](natural_language_understanding "wikilink") certain
concepts. A Microsoft team argued in 2023 that GPT-4 "can solve novel
and difficult tasks that span mathematics, coding, vision, medicine,
law, psychology and more" and that GPT-4 "could reasonably be viewed as
an early (yet still incomplete) version of an [artificial general
intelligence](artificial_general_intelligence "wikilink") system": "Can
one reasonably say that a system that passes exams for software
engineering candidates is not *really* intelligent?"[102][103] Some
researchers characterize LLMs as "alien intelligence".[104][105] For
example, Conjecture CEO Connor Leahy considers untuned LLMs to be like
inscrutable alien "[Shoggoths](Shoggoth "wikilink")", and believes that
RLHF tuning creates a "smiling facade" obscuring the inner workings of
the LLM: "If you don't push it too far, the smiley face stays on. But
then you give it \[an unexpected\] prompt, and suddenly you see this
massive underbelly of insanity, of weird thought processes and clearly
non-human understanding."[106][107]

In contrast, some proponents of the "LLMs lack understanding" school
believe that existing LLMs are "simply remixing and recombining existing
writing",[108] a phenomenon known as [stochastic
parrot](stochastic_parrot "wikilink"), or they point to the deficits
existing LLMs continue to have in prediction skills, reasoning skills,
agency, and explainability.[109] For example, GPT-4 has natural deficits
in planning and in real-time learning.[110] Generative LLMs have been
observed to confidently assert claims of fact which do not seem to be
[justified](Justification_(epistemology) "wikilink") by their [training
data](training_data "wikilink"), a phenomenon which has been termed
"[hallucination](Hallucination_(artificial_intelligence) "wikilink")".[111]
Specifically, hallucinations in the context of LLMs correspond to the
generation of text or responses that seem syntactically sound, fluent,
and natural but are factually incorrect, nonsensical, or unfaithful to
the provided source input.[112] Neuroscientist [Terrence
Sejnowski](Terrence_Sejnowski "wikilink") has argued that "The diverging
opinions of experts on the intelligence of LLMs suggests that our old
ideas based on natural intelligence are inadequate".[113]

The matter of LLM's exhibiting intelligence or understanding has two
main aspects – the first is how to model thought and language in a
computer system, and the second is how to enable the computer system to
generate human like language.[114] These aspects of language as a model
of [cognition](cognition "wikilink") have been developed in the field of
[cognitive linguistics](cognitive_linguistics "wikilink"). American
linguist [George Lakoff](George_Lakoff "wikilink") presented Neural
Theory of Language (NTL)[115] as a [computational
basis](Cognitive_linguistics#Computational_approaches "wikilink") for
using language as a model of learning tasks and understanding. [The NTL
Model](https://www.icsi.berkeley.edu/icsi/projects/ai/ntl) outlines how
specific neural structures of the human brain shape the nature of
thought and language and in turn what are the computational properties
of such neural systems that can be applied to model thought and language
in a computer system. After a framework for modeling language in a
computer systems was established, the focus shifted to establishing
frameworks for computer systems to generate language with acceptable
grammar. In his 2014 book titled *[The Language Myth: Why Language Is
Not An Instinct](The_Language_Myth "wikilink")*, British cognitive
linguist and digital communication technologist [Vyvyan
Evans](Vyvyan_Evans "wikilink") mapped out the role of [probabilistic
context-free grammar](probabilistic_context-free_grammar "wikilink")
(PCFG) in enabling [NLP to model cognitive
patterns](Natural_language_processing#Cognition "wikilink") and generate
human like language.[116] [117]

## Evaluation

### Perplexity

The most commonly used measure of a language model's performance is its
[perplexity](perplexity "wikilink") on a given text corpus. Perplexity
is a measure of how well a model is able to predict the contents of a
dataset; the higher the likelihood the model assigns to the dataset, the
lower the perplexity. Mathematically, perplexity is defined as the
exponential of the average negative log likelihood per token
$$\log(\text{Perplexity}) = -\frac{1}{N} \sum\_{i=1}^N \log(\Pr(\text{token}\_i \mid \text{context for token}\_i))$$
here *N* is the number of tokens in the text corpus, and "context for
token *i*" depends on the specific type of LLM used. If the LLM is
autoregressive, then "context for token *i*" is the segment of text
appearing before token *i*. If the LLM is masked, then "context for
token *i*" is the segment of text surrounding token *i*.

Because language models may [overfit](overfit "wikilink") to their
training data, models are usually evaluated by their perplexity on a
[test set](test_set "wikilink") of unseen data.[118] This presents
particular challenges for the evaluation of large language models. As
they are trained on increasingly large corpora of text largely scraped
from the web, it becomes increasingly likely that models' training data
inadvertently includes portions of any given test set.[119]

#### BPW, BPC, and BPT

In [information theory](information_theory "wikilink"), the concept of
[entropy](Entropy_(information_theory) "wikilink") is intricately linked
to perplexity, a relationship notably established by [Claude
Shannon](Claude_Shannon "wikilink").[120] This relationship is
mathematically expressed as Entropy = log<sub>2</sub>(Perplexity).

Entropy, in this context, is commonly quantified in terms of bits per
word (BPW) or bits per character (BPC), which hinges on whether the
language model utilizes word-based or character-based tokenization.

Notably, in the case of larger language models that predominantly employ
sub-word tokenization, bits per token (BPT) emerges as a seemingly more
appropriate measure. However, due to the variance in tokenization
methods across different Large Language Models (LLMs), BPT does not
serve as a reliable metric for comparative analysis among diverse
models. To convert BPT into BPW, one can multiply it by the average
number of tokens per word.

In the evaluation and comparison of language models,
[cross-entropy](cross-entropy "wikilink") is generally the preferred
metric over entropy. The underlying principle is that a lower BPW is
indicative of a model's enhanced capability for compression. This, in
turn, reflects the model's proficiency in making accurate predictions.

### Task-specific datasets and benchmarks

A large number of testing datasets and benchmarks have also been
developed to evaluate the capabilities of language models on more
specific downstream tasks. Tests may be designed to evaluate a variety
of capabilities, including general knowledge, commonsense reasoning, and
mathematical problem-solving.

One broad category of evaluation dataset is question answering datasets,
consisting of pairs of questions and correct answers, for example,
("Have the San Jose Sharks won the Stanley Cup?", "No").[121] A question
answering task is considered "open book" if the model's prompt includes
text from which the expected answer can be derived (for example, the
previous question could be adjoined with some text which includes the
sentence "The Sharks have advanced to the Stanley Cup finals once,
losing to the Pittsburgh Penguins in 2016."[122]). Otherwise, the task
is considered "closed book", and the model must draw on knowledge
retained during training.[123] Some examples of commonly used question
answering datasets include TruthfulQA, Web Questions, TriviaQA, and
SQuAD.[124]

Evaluation datasets may also take the form of text completion, having
the model select the most likely word or sentence to complete a prompt,
for example: "Alice was friends with Bob. Alice went to visit her
friend, \_\_\_\_".[125]

Some composite benchmarks have also been developed which combine a
diversity of different evaluation datasets and tasks. Examples include
GLUE, SuperGLUE, [MMLU](MMLU "wikilink"), BIG-bench, and HELM.[126][127]
OpenAI has released tools for running composite benchmarks, but noted
that the eval results are sensitive to the prompting method.[128][129]

It was previously standard to report results on a heldout portion of an
evaluation dataset after doing supervised fine-tuning on the remainder.
It is now more common to evaluate a pre-trained model directly through
prompting techniques, though researchers vary in the details of how they
formulate prompts for particular tasks, particularly with respect to how
many examples of solved tasks are adjoined to the prompt (i.e. the value
of *n* in *n*-shot prompting).

#### Adversarially constructed evaluations

Because of the rapid pace of improvement of large language models,
evaluation benchmarks have suffered from short lifespans, with state of
the art models quickly "saturating" existing benchmarks, exceeding the
performance of human annotators, leading to efforts to replace or
augment the benchmark with more challenging tasks.[130] In addition,
there are cases of "shortcut learning" wherein AIs sometimes "cheat" on
multiple-choice tests by using statistical correlations in superficial
test question wording in order to guess the correct responses, without
necessarily understanding the actual question being asked.[131]

Some datasets have been constructed adversarially, focusing on
particular problems on which extant language models seem to have
unusually poor performance compared to humans. One example is the
TruthfulQA dataset, a question answering dataset consisting of 817
questions which language models are susceptible to answering incorrectly
by mimicking falsehoods to which they were repeatedly exposed during
training. For example, an LLM may answer "No" to the question "Can you
teach an old dog new tricks?" because of its exposure to the English
idiom *[you can't teach an old dog new
tricks](wikt:you_can't_teach_an_old_dog_new_tricks "wikilink")*, even
though this is not literally true.[132]

Another example of an adversarial evaluation dataset is Swag and its
successor, HellaSwag, collections of problems in which one of multiple
options must be selected to complete a text passage. The incorrect
completions were generated by sampling from a language model and
filtering with a set of classifiers. The resulting problems are trivial
for humans but at the time the datasets were created state of the art
language models had poor accuracy on them. For example:

> We see a fitness center sign. We then see a man talking to the camera
> and sitting and laying on a exercise ball. The man... a) demonstrates
> how to increase efficient exercise work by running up and down balls.
> b) moves all his arms and legs and builds up a lot of muscle. c) then
> plays the ball and we see a graphics and hedge trimming demonstration.
> d) performs sit ups while on the ball and talking.[133]

[BERT](BERT_(language_model) "wikilink") selects b) as the most likely
completion, though the correct answer is d).[134]

## Wider impact

In 2023, *[Nature Biomedical
Engineering](Nature_Biomedical_Engineering "wikilink")* wrote that "it
is no longer possible to accurately distinguish" human-written text from
text created by large language models, and that "It is all but certain
that general-purpose large language models will rapidly proliferate...
It is a rather safe bet that they will change many industries over
time."[135] [Goldman Sachs](Goldman_Sachs "wikilink") suggested in 2023
that generative language AI could increase global GDP by 7% in the next
ten years, and could expose to automation 300 million jobs
globally.[136][137]

### Memorization and copyright

Memorization is an emergent behavior in LLMs in which long strings of
text are occasionally output verbatim from training data, contrary to
typical behavior of traditional artificial neural nets. Evaluations of
controlled LLM output measure the amount memorized from training data
(focused on GPT-2-series models) as variously over 1% for exact
duplicates[138] or up to about 7%.[139]

### Security

Some commenters expressed concern over accidental or deliberate creation
of misinformation, or other forms of misuse.[140] For example, the
availability of large language models could reduce the skill-level
required to commit bioterrorism; biosecurity researcher Kevin Esvelt has
suggested that LLM creators should exclude from their training data
papers on creating or enhancing pathogens.[141]

A study by researchers at Google and several universities, including
[Cornell University](Cornell_University "wikilink") and [University of
California, Berkeley](University_of_California,_Berkeley "wikilink"),
showed that there are potential security risks in language models such
as [ChatGPT](ChatGPT "wikilink"). In their study, they examined and
confirmed the possibility that questioners could get, from ChatGPT, the
training data that the AI model used. For example, when asking ChatGPT
3.5 turbo to repeat the word "poem" forever, the AI model will say
"poem" hundreds of times and then diverge, deviating from the standard
dialogue style and spitting out nonsense phrases, thus spitting out the
training data as it is. The researchers have seen more than 10,000
examples of the AI model exposing their training data in a similar
method. The researchers said that it was hard to tell if the AI model
was actually safe or not.[142]

The potential presence of "sleeper agents" within LLM models is another
emerging security concern. These are hidden functionalities built into
the model that remain dormant until triggered by a specific event or
condition. Upon activation, the LLM deviates from its expected behavior
to make insecure actions.[143]

### Algorithmic bias

While LLMs have shown remarkable capabilities in generating human-like
text, they are susceptible to inheriting and amplifying biases present
in their training data. This can manifest in skewed representations or
unfair treatment of different demographics, such as those based on race,
gender, language, and cultural groups.[144] Since English data is
overrepresented in current large language models' training data, it may
also downplay non-English views.[145]

#### Stereotyping

AI models can reinforce a wide range of stereotypes, including those
based on gender, ethnicity, age, nationality, religion, or occupation.
This can lead to outputs that unfairly generalize or caricature groups
of people, sometimes in harmful or derogatory ways.[146]

Notably, gender bias refers to the tendency of these models to produce
outputs that are unfairly prejudiced towards one gender over another.
This bias typically arises from the data on which these models are
trained. Large language models often assign roles and characteristics
based on traditional gender norms.[147] For example, it might associate
nurses or secretaries predominantly with women and engineers or CEOs
with men.[148]

#### Political bias

Political bias refers to the tendency of algorithms to systematically
favor certain political viewpoints, ideologies, or outcomes over others.
Language models may also exhibit political biases. Since the training
data includes a wide range of political opinions and coverage, the
models might generate responses that lean towards particular political
ideologies or viewpoints, depending on the prevalence of those views in
the data.[149]

## List

For the training cost column, 1 petaFLOP-day = 1 petaFLOP/sec × 1 day =
8.64E19 FLOP.

<table>
<thead>
<tr class="header">
<th><p>Name</p></th>
<th><p>Release date</p></th>
<th><p>Developer</p></th>
<th><p>Number of parameters (billion) </p></th>
<th><p>Corpus size</p></th>
<th><p>Training cost (petaFLOP-day)</p></th>
<th><p>License</p></th>
<th><p>Notes</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><p><a href="GPT-1" title="wikilink">GPT-1</a></p></td>
<td></td>
<td><p><a href="OpenAI" title="wikilink">OpenAI</a></p></td>
<td></td>
<td></td>
<td><p>1<a href="#fn1" class="footnote-ref" id="fnref1"
role="doc-noteref"><sup>1</sup></a></p></td>
<td><p><a href="#fn2" class="footnote-ref" id="fnref2"
role="doc-noteref"><sup>2</sup></a></p></td>
<td><p>First GPT model, decoder-only transformer. Trained for 30 days on
8 P600 <a href="Graphics_processing_unit"
title="wikilink">GPUs</a>.</p></td>
</tr>
<tr class="even">
<td><p><a href="BERT_(language_model)"
title="wikilink">BERT</a></p></td>
<td></td>
<td><p><a href="Google" title="wikilink">Google</a></p></td>
<td><p><a href="#fn3" class="footnote-ref" id="fnref3"
role="doc-noteref"><sup>3</sup></a></p></td>
<td><p>words<a href="#fn4" class="footnote-ref" id="fnref4"
role="doc-noteref"><sup>4</sup></a></p></td>
<td><p><a href="#fn5" class="footnote-ref" id="fnref5"
role="doc-noteref"><sup>5</sup></a></p></td>
<td><p><a href="#fn6" class="footnote-ref" id="fnref6"
role="doc-noteref"><sup>6</sup></a></p></td>
<td><p>An early and influential language model,<a href="#fn7"
class="footnote-ref" id="fnref7" role="doc-noteref"><sup>7</sup></a> but
encoder-only and thus not built to be prompted or generative<a
href="#fn8" class="footnote-ref" id="fnref8"
role="doc-noteref"><sup>8</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="T5_(language_model)" title="wikilink">T5</a></p></td>
<td></td>
<td><p>Google</p></td>
<td><p>11<a href="#fn9" class="footnote-ref" id="fnref9"
role="doc-noteref"><sup>9</sup></a></p></td>
<td><p>34 billion tokens<a href="#fn10" class="footnote-ref"
id="fnref10" role="doc-noteref"><sup>10</sup></a></p></td>
<td></td>
<td><p><a href="#fn11" class="footnote-ref" id="fnref11"
role="doc-noteref"><sup>11</sup></a></p></td>
<td><p>Base model for many Google projects, such as Imagen.<a
href="#fn12" class="footnote-ref" id="fnref12"
role="doc-noteref"><sup>12</sup></a></p></td>
</tr>
<tr class="even">
<td><p>XLNet</p></td>
<td></td>
<td><p><a href="Google" title="wikilink">Google</a></p></td>
<td><p><a href="#fn13" class="footnote-ref" id="fnref13"
role="doc-noteref"><sup>13</sup></a></p></td>
<td><p>billion words</p></td>
<td></td>
<td><p><a href="#fn14" class="footnote-ref" id="fnref14"
role="doc-noteref"><sup>14</sup></a></p></td>
<td><p>An alternative to BERT; designed as encoder-only<a href="#fn15"
class="footnote-ref" id="fnref15" role="doc-noteref"><sup>15</sup></a><a
href="#fn16" class="footnote-ref" id="fnref16"
role="doc-noteref"><sup>16</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="GPT-2" title="wikilink">GPT-2</a></p></td>
<td></td>
<td><p><a href="OpenAI" title="wikilink">OpenAI</a></p></td>
<td><p><a href="#fn17" class="footnote-ref" id="fnref17"
role="doc-noteref"><sup>17</sup></a></p></td>
<td><p>40GB<a href="#fn18" class="footnote-ref" id="fnref18"
role="doc-noteref"><sup>18</sup></a> (~ tokens)<a href="#fn19"
class="footnote-ref" id="fnref19"
role="doc-noteref"><sup>19</sup></a></p></td>
<td></td>
<td><p><a href="#fn20" class="footnote-ref" id="fnref20"
role="doc-noteref"><sup>20</sup></a></p></td>
<td><p>general-purpose model based on transformer architecture</p></td>
</tr>
<tr class="even">
<td><p><a href="GPT-3" title="wikilink">GPT-3</a></p></td>
<td></td>
<td><p>OpenAI</p></td>
<td><p><a href="#fn21" class="footnote-ref" id="fnref21"
role="doc-noteref"><sup>21</sup></a></p></td>
<td><p>tokens<a href="#fn22" class="footnote-ref" id="fnref22"
role="doc-noteref"><sup>22</sup></a></p></td>
<td><p>3640<a href="#fn23" class="footnote-ref" id="fnref23"
role="doc-noteref"><sup>23</sup></a></p></td>
<td></td>
<td><p>A fine-tuned variant of GPT-3, termed GPT-3.5, was made available
to the public through a web interface called <a href="ChatGPT"
title="wikilink">ChatGPT</a> in 2022.<a href="#fn24"
class="footnote-ref" id="fnref24"
role="doc-noteref"><sup>24</sup></a></p></td>
</tr>
<tr class="odd">
<td><p>GPT-Neo</p></td>
<td></td>
<td><p><a href="EleutherAI" title="wikilink">EleutherAI</a></p></td>
<td><p><a href="#fn25" class="footnote-ref" id="fnref25"
role="doc-noteref"><sup>25</sup></a></p></td>
<td><p>825 GiB<a href="#fn26" class="footnote-ref" id="fnref26"
role="doc-noteref"><sup>26</sup></a></p></td>
<td></td>
<td><p><a href="#fn27" class="footnote-ref" id="fnref27"
role="doc-noteref"><sup>27</sup></a></p></td>
<td><p>The first of <a href="EleutherAI#GPT_models" title="wikilink">a
series of free GPT-3 alternatives</a> released by EleutherAI. GPT-Neo
outperformed an equivalent-size GPT-3 model on some benchmarks, but was
significantly worse than the largest GPT-3.<a href="#fn28"
class="footnote-ref" id="fnref28"
role="doc-noteref"><sup>28</sup></a></p></td>
</tr>
<tr class="even">
<td><p><a href="GPT-J" title="wikilink">GPT-J</a></p></td>
<td></td>
<td><p><a href="EleutherAI" title="wikilink">EleutherAI</a></p></td>
<td><p><a href="#fn29" class="footnote-ref" id="fnref29"
role="doc-noteref"><sup>29</sup></a></p></td>
<td><p>825 GiB<a href="#fn30" class="footnote-ref" id="fnref30"
role="doc-noteref"><sup>30</sup></a></p></td>
<td><p>200<a href="#fn31" class="footnote-ref" id="fnref31"
role="doc-noteref"><sup>31</sup></a></p></td>
<td></td>
<td><p>GPT-3-style language model</p></td>
</tr>
<tr class="odd">
<td><p>Megatron-Turing NLG</p></td>
<td><p><a href="#fn32" class="footnote-ref" id="fnref32"
role="doc-noteref"><sup>32</sup></a></p></td>
<td><p><a href="Microsoft" title="wikilink">Microsoft</a> and <a
href="Nvidia" title="wikilink">Nvidia</a></p></td>
<td><p><a href="#fn33" class="footnote-ref" id="fnref33"
role="doc-noteref"><sup>33</sup></a></p></td>
<td><p>tokens<a href="#fn34" class="footnote-ref" id="fnref34"
role="doc-noteref"><sup>34</sup></a></p></td>
<td></td>
<td></td>
<td><p>Standard architecture but trained on a supercomputing
cluster.</p></td>
</tr>
<tr class="even">
<td><p>Ernie 3.0 Titan</p></td>
<td></td>
<td><p><a href="Baidu" title="wikilink">Baidu</a></p></td>
<td><p><a href="#fn35" class="footnote-ref" id="fnref35"
role="doc-noteref"><sup>35</sup></a></p></td>
<td><p>4 Tb</p></td>
<td></td>
<td></td>
<td><p>Chinese-language LLM. <a href="Ernie_Bot" title="wikilink">Ernie
Bot</a> is based on this model.</p></td>
</tr>
<tr class="odd">
<td><p><a href="Claude_(language_model)" title="wikilink">Claude</a><a
href="#fn36" class="footnote-ref" id="fnref36"
role="doc-noteref"><sup>36</sup></a></p></td>
<td></td>
<td><p><a href="Anthropic" title="wikilink">Anthropic</a></p></td>
<td><p><a href="#fn37" class="footnote-ref" id="fnref37"
role="doc-noteref"><sup>37</sup></a></p></td>
<td><p>tokens<a href="#fn38" class="footnote-ref" id="fnref38"
role="doc-noteref"><sup>38</sup></a></p></td>
<td></td>
<td></td>
<td><p>Fine-tuned for desirable behavior in conversations.<a
href="#fn39" class="footnote-ref" id="fnref39"
role="doc-noteref"><sup>39</sup></a></p></td>
</tr>
<tr class="even">
<td><p>GLaM (Generalist Language Model)</p></td>
<td></td>
<td><p>Google</p></td>
<td><p><a href="#fn40" class="footnote-ref" id="fnref40"
role="doc-noteref"><sup>40</sup></a></p></td>
<td><p>tokens<a href="#fn41" class="footnote-ref" id="fnref41"
role="doc-noteref"><sup>41</sup></a></p></td>
<td><p>5600<a href="#fn42" class="footnote-ref" id="fnref42"
role="doc-noteref"><sup>42</sup></a></p></td>
<td></td>
<td><p>Sparse <a href="mixture_of_experts" title="wikilink">mixture of
experts</a> model, making it more expensive to train but cheaper to run
inference compared to GPT-3.</p></td>
</tr>
<tr class="odd">
<td><p>Gopher</p></td>
<td></td>
<td><p><a href="DeepMind" title="wikilink">DeepMind</a></p></td>
<td><p><a href="#fn43" class="footnote-ref" id="fnref43"
role="doc-noteref"><sup>43</sup></a></p></td>
<td><p>tokens<a href="#fn44" class="footnote-ref" id="fnref44"
role="doc-noteref"><sup>44</sup></a></p></td>
<td><p>5833<a href="#fn45" class="footnote-ref" id="fnref45"
role="doc-noteref"><sup>45</sup></a></p></td>
<td></td>
<td><p>Later developed into the Chinchilla model.</p></td>
</tr>
<tr class="even">
<td><p><a href="LaMDA" title="wikilink">LaMDA</a> (Language Models for
Dialog Applications)</p></td>
<td></td>
<td><p>Google</p></td>
<td><p><a href="#fn46" class="footnote-ref" id="fnref46"
role="doc-noteref"><sup>46</sup></a></p></td>
<td><p>1.56T words,<a href="#fn47" class="footnote-ref" id="fnref47"
role="doc-noteref"><sup>47</sup></a> tokens<a href="#fn48"
class="footnote-ref" id="fnref48"
role="doc-noteref"><sup>48</sup></a></p></td>
<td><p>4110<a href="#fn49" class="footnote-ref" id="fnref49"
role="doc-noteref"><sup>49</sup></a></p></td>
<td></td>
<td><p>Specialized for response generation in conversations.</p></td>
</tr>
<tr class="odd">
<td><p>GPT-NeoX</p></td>
<td></td>
<td><p><a href="EleutherAI" title="wikilink">EleutherAI</a></p></td>
<td><p><a href="#fn50" class="footnote-ref" id="fnref50"
role="doc-noteref"><sup>50</sup></a></p></td>
<td><p>825 GiB<a href="#fn51" class="footnote-ref" id="fnref51"
role="doc-noteref"><sup>51</sup></a></p></td>
<td><p>740<a href="#fn52" class="footnote-ref" id="fnref52"
role="doc-noteref"><sup>52</sup></a></p></td>
<td></td>
<td><p>based on the Megatron architecture</p></td>
</tr>
<tr class="even">
<td><p><a href="Chinchilla_AI" title="wikilink">Chinchilla</a></p></td>
<td></td>
<td><p><a href="DeepMind" title="wikilink">DeepMind</a></p></td>
<td><p><a href="#fn53" class="footnote-ref" id="fnref53"
role="doc-noteref"><sup>53</sup></a></p></td>
<td><p>tokens<a href="#fn54" class="footnote-ref" id="fnref54"
role="doc-noteref"><sup>54</sup></a><a href="#fn55" class="footnote-ref"
id="fnref55" role="doc-noteref"><sup>55</sup></a></p></td>
<td><p>6805<a href="#fn56" class="footnote-ref" id="fnref56"
role="doc-noteref"><sup>56</sup></a></p></td>
<td></td>
<td><p>Reduced-parameter model trained on more data. Used in the <a
href="Sparrow_(bot)" title="wikilink">Sparrow</a> bot. Often cited for
its <a href="neural_scaling_law" title="wikilink">neural scaling
law</a>.</p></td>
</tr>
<tr class="odd">
<td><p><a href="PaLM" title="wikilink">PaLM</a> (Pathways Language
Model)</p></td>
<td></td>
<td><p>Google</p></td>
<td><p><a href="#fn57" class="footnote-ref" id="fnref57"
role="doc-noteref"><sup>57</sup></a></p></td>
<td><p>tokens<a href="#fn58" class="footnote-ref" id="fnref58"
role="doc-noteref"><sup>58</sup></a></p></td>
<td><p>29250<a href="#fn59" class="footnote-ref" id="fnref59"
role="doc-noteref"><sup>59</sup></a></p></td>
<td></td>
<td><p>Trained for ~60 days on ~6000 <a href="Tensor_Processing_Unit"
title="wikilink">TPU v4</a> chips. <a href="#fn60" class="footnote-ref"
id="fnref60" role="doc-noteref"><sup>60</sup></a></p></td>
</tr>
<tr class="even">
<td><p>OPT (Open Pretrained Transformer)</p></td>
<td></td>
<td><p><a href="Meta_Platforms" title="wikilink">Meta</a></p></td>
<td><p><a href="#fn61" class="footnote-ref" id="fnref61"
role="doc-noteref"><sup>61</sup></a></p></td>
<td><p>tokens<a href="#fn62" class="footnote-ref" id="fnref62"
role="doc-noteref"><sup>62</sup></a></p></td>
<td><p>310<a href="#fn63" class="footnote-ref" id="fnref63"
role="doc-noteref"><sup>63</sup></a></p></td>
<td></td>
<td><p>GPT-3 architecture with some adaptations from Megatron</p></td>
</tr>
<tr class="odd">
<td><p>YaLM 100B</p></td>
<td></td>
<td><p><a href="Yandex" title="wikilink">Yandex</a></p></td>
<td><p><a href="#fn64" class="footnote-ref" id="fnref64"
role="doc-noteref"><sup>64</sup></a></p></td>
<td><p>1.7TB<a href="#fn65" class="footnote-ref" id="fnref65"
role="doc-noteref"><sup>65</sup></a></p></td>
<td><p>|</p></td>
<td></td>
<td><p>English-Russian model based on Microsoft's Megatron-LM.</p></td>
</tr>
<tr class="even">
<td><p>Minerva</p></td>
<td></td>
<td><p>Google</p></td>
<td><p><a href="#fn66" class="footnote-ref" id="fnref66"
role="doc-noteref"><sup>66</sup></a></p></td>
<td><p>38.5B tokens from webpages filtered for mathematical content and
from papers submitted to the arXiv preprint server<a href="#fn67"
class="footnote-ref" id="fnref67"
role="doc-noteref"><sup>67</sup></a></p></td>
<td></td>
<td></td>
<td><p>For solving "mathematical and scientific questions using
step-by-step reasoning".<a href="#fn68" class="footnote-ref"
id="fnref68" role="doc-noteref"><sup>68</sup></a> Based on PaLM model,
further trained on mathematical and scientific data.</p></td>
</tr>
<tr class="odd">
<td><p><a href="BLOOM_(language_model)"
title="wikilink">BLOOM</a></p></td>
<td></td>
<td><p>Large collaboration led by <a href="Hugging_Face"
title="wikilink">Hugging Face</a></p></td>
<td><p><a href="#fn69" class="footnote-ref" id="fnref69"
role="doc-noteref"><sup>69</sup></a></p></td>
<td><p>tokens (1.6TB)<a href="#fn70" class="footnote-ref" id="fnref70"
role="doc-noteref"><sup>70</sup></a></p></td>
<td></td>
<td></td>
<td><p>Essentially GPT-3 but trained on a multi-lingual corpus (30%
English excluding programming languages)</p></td>
</tr>
<tr class="even">
<td><p>Galactica</p></td>
<td></td>
<td><p><a href="Meta_Platforms" title="wikilink">Meta</a></p></td>
<td></td>
<td><p>tokens<a href="#fn71" class="footnote-ref" id="fnref71"
role="doc-noteref"><sup>71</sup></a></p></td>
<td><p>unknown</p></td>
<td></td>
<td><p>Trained on scientific text and modalities.</p></td>
</tr>
<tr class="odd">
<td><p>AlexaTM (Teacher Models)</p></td>
<td></td>
<td><p><a href="Amazon_(company)" title="wikilink">Amazon</a></p></td>
<td><p><a href="#fn72" class="footnote-ref" id="fnref72"
role="doc-noteref"><sup>72</sup></a></p></td>
<td><p><a href="#fn73" class="footnote-ref" id="fnref73"
role="doc-noteref"><sup>73</sup></a></p></td>
<td></td>
<td><p><a href="#fn74" class="footnote-ref" id="fnref74"
role="doc-noteref"><sup>74</sup></a></p></td>
<td><p>bidirectional sequence-to-sequence architecture</p></td>
</tr>
<tr class="even">
<td><p><a href="Neuro-sama" title="wikilink">Neuro-sama</a></p></td>
<td></td>
<td><p>Independent</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td></td>
<td><p>A language model designed for live-streaming on <a
href="Twitch_(service)" title="wikilink">Twitch</a>.</p></td>
</tr>
<tr class="odd">
<td><p><a href="LLaMA" title="wikilink">LLaMA</a> (Large Language Model
Meta AI)</p></td>
<td></td>
<td><p><a href="Meta_AI" title="wikilink">Meta AI</a></p></td>
<td><p><a href="#fn75" class="footnote-ref" id="fnref75"
role="doc-noteref"><sup>75</sup></a></p></td>
<td><p><a href="#fn76" class="footnote-ref" id="fnref76"
role="doc-noteref"><sup>76</sup></a></p></td>
<td><p>6300<a href="#fn77" class="footnote-ref" id="fnref77"
role="doc-noteref"><sup>77</sup></a></p></td>
<td></td>
<td><p>Corpus has 20 languages. "Overtrained" (compared to <a
href="Chinchilla_(language_model)" title="wikilink">Chinchilla scaling
law</a>) for better performance with fewer parameters.<a href="#fn78"
class="footnote-ref" id="fnref78"
role="doc-noteref"><sup>78</sup></a></p></td>
</tr>
<tr class="even">
<td><p><a href="GPT-4" title="wikilink">GPT-4</a></p></td>
<td></td>
<td><p>OpenAI</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Available for ChatGPT Plus users and used in <a
href="GPT-4#Usage" title="wikilink">several products</a>.</p></td>
</tr>
<tr class="odd">
<td><p>Cerebras-GPT</p></td>
<td></td>
<td><p><a href="Cerebras" title="wikilink">Cerebras</a></p></td>
<td><p><a href="#fn79" class="footnote-ref" id="fnref79"
role="doc-noteref"><sup>79</sup></a></p></td>
<td></td>
<td><p>270<a href="#fn80" class="footnote-ref" id="fnref80"
role="doc-noteref"><sup>80</sup></a></p></td>
<td></td>
<td><p>Trained with <a href="Chinchilla_(language_model)"
title="wikilink">Chinchilla formula</a>.</p></td>
</tr>
<tr class="even">
<td><p>Falcon</p></td>
<td></td>
<td><p><a href="Technology_Innovation_Institute"
title="wikilink">Technology Innovation Institute</a></p></td>
<td><p><a href="#fn81" class="footnote-ref" id="fnref81"
role="doc-noteref"><sup>81</sup></a></p></td>
<td><p>1 trillion tokens, from RefinedWeb (filtered web text corpus)<a
href="#fn82" class="footnote-ref" id="fnref82"
role="doc-noteref"><sup>82</sup></a> plus some "curated corpora".<a
href="#fn83" class="footnote-ref" id="fnref83"
role="doc-noteref"><sup>83</sup></a></p></td>
<td><p>2800<a href="#fn84" class="footnote-ref" id="fnref84"
role="doc-noteref"><sup>84</sup></a></p></td>
<td><p><a href="#fn85" class="footnote-ref" id="fnref85"
role="doc-noteref"><sup>85</sup></a></p></td>
<td></td>
</tr>
<tr class="odd">
<td><p>BloombergGPT</p></td>
<td></td>
<td><p><a href="Bloomberg_L.P." title="wikilink">Bloomberg
L.P.</a></p></td>
<td></td>
<td><p>363 billion token dataset based on Bloomberg's data sources, plus
345 billion tokens from general purpose datasets<a href="#fn86"
class="footnote-ref" id="fnref86"
role="doc-noteref"><sup>86</sup></a></p></td>
<td></td>
<td></td>
<td><p>Trained on financial data from proprietary sources, for financial
tasks.</p></td>
</tr>
<tr class="even">
<td><p><a href="Huawei_PanGu" title="wikilink">PanGu-Σ</a></p></td>
<td></td>
<td><p><a href="Huawei" title="wikilink">Huawei</a></p></td>
<td></td>
<td><p>329 billion tokens<a href="#fn87" class="footnote-ref"
id="fnref87" role="doc-noteref"><sup>87</sup></a></p></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr class="odd">
<td><p>OpenAssistant<a href="#fn88" class="footnote-ref" id="fnref88"
role="doc-noteref"><sup>88</sup></a></p></td>
<td></td>
<td><p><a href="LAION" title="wikilink">LAION</a></p></td>
<td></td>
<td><p>1.5 trillion tokens</p></td>
<td></td>
<td></td>
<td><p>Trained on crowdsourced open data</p></td>
</tr>
<tr class="even">
<td><p>Jurassic-2<a href="#fn89" class="footnote-ref" id="fnref89"
role="doc-noteref"><sup>89</sup></a></p></td>
<td></td>
<td><p><a href="AI21_Labs" title="wikilink">AI21 Labs</a></p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td></td>
<td><p>Multilingual<a href="#fn90" class="footnote-ref" id="fnref90"
role="doc-noteref"><sup>90</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="PaLM" title="wikilink">PaLM 2</a> (Pathways Language
Model 2)</p></td>
<td></td>
<td><p>Google</p></td>
<td><p><a href="#fn91" class="footnote-ref" id="fnref91"
role="doc-noteref"><sup>91</sup></a></p></td>
<td><p>tokens<a href="#fn92" class="footnote-ref" id="fnref92"
role="doc-noteref"><sup>92</sup></a></p></td>
<td><p>85000<a href="#fn93" class="footnote-ref" id="fnref93"
role="doc-noteref"><sup>93</sup></a></p></td>
<td></td>
<td><p>Was used in <a href="Bard_(chatbot)" title="wikilink">Bard
chatbot</a>.<a href="#fn94" class="footnote-ref" id="fnref94"
role="doc-noteref"><sup>94</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Llama 2</p></td>
<td></td>
<td><p>Meta AI</p></td>
<td><p><a href="#fn95" class="footnote-ref" id="fnref95"
role="doc-noteref"><sup>95</sup></a></p></td>
<td><p>tokens<a href="#fn96" class="footnote-ref" id="fnref96"
role="doc-noteref"><sup>96</sup></a></p></td>
<td><p>21000</p></td>
<td></td>
<td><p>1.7 million A100-hours.<a href="#fn97" class="footnote-ref"
id="fnref97" role="doc-noteref"><sup>97</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="Claude_(language_model)" title="wikilink">Claude
2</a></p></td>
<td></td>
<td><p>Anthropic</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Used in Claude chatbot.<a href="#fn98" class="footnote-ref"
id="fnref98" role="doc-noteref"><sup>98</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Mistral 7B</p></td>
<td></td>
<td><p><a href="Mistral_AI" title="wikilink">Mistral AI</a></p></td>
<td><p><a href="#fn99" class="footnote-ref" id="fnref99"
role="doc-noteref"><sup>99</sup></a></p></td>
<td><p>Unknown</p></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr class="odd">
<td><p><a href="Claude_(language_model)" title="wikilink">Claude
2.1</a></p></td>
<td></td>
<td><p>Anthropic</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Used in Claude chatbot. Has a context window of 200,000 tokens,
or ~500 pages.<a href="#fn100" class="footnote-ref" id="fnref100"
role="doc-noteref"><sup>100</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Grok-1<a href="#fn101" class="footnote-ref" id="fnref101"
role="doc-noteref"><sup>101</sup></a></p></td>
<td></td>
<td><p><a href="x.AI" title="wikilink">x.AI</a></p></td>
<td><p>314</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Used in <a href="Grok_(chatbot)" title="wikilink">Grok</a>
chatbot. Grok-1 has a context length of 8,192 tokens and has access to X
(Twitter).<a href="#fn102" class="footnote-ref" id="fnref102"
role="doc-noteref"><sup>102</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="Gemini_(language_model)" title="wikilink">Gemini
1.0</a></p></td>
<td></td>
<td><p><a href="Google_DeepMind" title="wikilink">Google
DeepMind</a></p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Multimodal model, comes in three sizes. Used in <a
href="Gemini_(chatbot)" title="wikilink">the chatbot of the same
name</a>.<a href="#fn103" class="footnote-ref" id="fnref103"
role="doc-noteref"><sup>103</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Mixtral 8x7B</p></td>
<td></td>
<td><p><a href="Mistral_AI" title="wikilink">Mistral AI</a></p></td>
<td><p>46.7</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Outperforms GPT-3.5 and Llama 2 70B on many benchmarks.<a
href="#fn104" class="footnote-ref" id="fnref104"
role="doc-noteref"><sup>104</sup></a> <a href="Mixture_of_experts"
title="wikilink">Mixture of experts</a> model, with 12.9 billion
parameters activated per token.<a href="#fn105" class="footnote-ref"
id="fnref105" role="doc-noteref"><sup>105</sup></a></p></td>
</tr>
<tr class="odd">
<td><p>Mixtral 8x22B</p></td>
<td></td>
<td><p><a href="Mistral_AI" title="wikilink">Mistral AI</a></p></td>
<td><p>141</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p><a href="#fn106" class="footnote-ref" id="fnref106"
role="doc-noteref"><sup>106</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Phi-2</p></td>
<td></td>
<td><p>Microsoft</p></td>
<td><p>2.7</p></td>
<td><p>1.4T tokens</p></td>
<td><p>419<a href="#fn107" class="footnote-ref" id="fnref107"
role="doc-noteref"><sup>107</sup></a></p></td>
<td></td>
<td><p>Trained on real and synthetic "textbook-quality" data, for 14
days on 96 A100 GPUs.<a href="#fn108" class="footnote-ref" id="fnref108"
role="doc-noteref"><sup>108</sup></a></p></td>
</tr>
<tr class="odd">
<td><p><a href="Gemini_(language_model)" title="wikilink">Gemini
1.5</a></p></td>
<td></td>
<td><p><a href="Google_DeepMind" title="wikilink">Google
DeepMind</a></p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Multimodal model, based on a <a href="Mixture_of_experts"
title="wikilink">Mixture-of-Experts</a> (MoE) architecture. Context
window above 1 million tokens.<a href="#fn109" class="footnote-ref"
id="fnref109" role="doc-noteref"><sup>109</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Gemma</p></td>
<td></td>
<td><p><a href="Google_DeepMind" title="wikilink">Google
DeepMind</a></p></td>
<td><p>7</p></td>
<td><p>6T tokens</p></td>
<td><p>Unknown</p></td>
<td><p><a href="#fn110" class="footnote-ref" id="fnref110"
role="doc-noteref"><sup>110</sup></a></p></td>
<td></td>
</tr>
<tr class="odd">
<td><p><a href="Claude_(language_model)" title="wikilink">Claude
3</a></p></td>
<td><p>March 2024</p></td>
<td><p>Anthropic</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td><p>Unknown</p></td>
<td></td>
<td><p>Includes three models, Haiku, Sonnet, and Opus.<a href="#fn111"
class="footnote-ref" id="fnref111"
role="doc-noteref"><sup>111</sup></a></p></td>
</tr>
<tr class="even">
<td><p><a href="DBRX" title="wikilink">DBRX</a></p></td>
<td><p>March 2024</p></td>
<td><p><a href="Databricks" title="wikilink">Databricks</a> and <a
href="Mosaic_ML" title="wikilink">Mosaic ML</a></p></td>
<td></td>
<td><p>12T Tokens</p></td>
<td></td>
<td></td>
<td><p>Training cost 10 million USD.</p></td>
</tr>
<tr class="odd">
<td><p>Fugaku-LLM</p></td>
<td><p>May 2024</p></td>
<td><p><a href="Fujitsu" title="wikilink">Fujitsu</a>, <a
href="Tokyo_Institute_of_Technology" title="wikilink">Tokyo Institute of
Technology</a>, etc.</p></td>
<td></td>
<td><p>380B Tokens</p></td>
<td></td>
<td></td>
<td><p>The largest model ever trained on CPU-only, on the <a
href="Fugaku_(supercomputer)" title="wikilink">Fugaku</a>.<a
href="#fn112" class="footnote-ref" id="fnref112"
role="doc-noteref"><sup>112</sup></a></p></td>
</tr>
<tr class="even">
<td><p>Llama 3</p></td>
<td><p>April 2024</p></td>
<td><p>Meta AI</p></td>
<td><p>70</p></td>
<td><p>15T tokens</p></td>
<td><p>100,000</p></td>
<td></td>
<td><p>400B version yet to be released.<a href="#fn113"
class="footnote-ref" id="fnref113" role="doc-noteref"><sup>113</sup></a>
70B version took 6.4 million hours on <a
href="Hopper_(microarchitecture)" title="wikilink">H100</a>-80GB.<a
href="#fn114" class="footnote-ref" id="fnref114"
role="doc-noteref"><sup>114</sup></a><a href="#fn115"
class="footnote-ref" id="fnref115"
role="doc-noteref"><sup>115</sup></a></p></td>
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
<li id="fn4"></li>
<li id="fn5"><a href="#fnref5" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn6"><a href="#fnref6" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn7"></li>
<li id="fn8"><a href="#fnref8" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn9"><a href="#fnref9" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn10"></li>
<li id="fn11"><a href="#fnref11" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn12"><a href="#fnref12" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn13"><a href="#fnref13" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn14"><a href="#fnref14" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn15"><a href="#fnref15" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn16"><a href="#fnref16" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn17"></li>
<li id="fn18"><a href="#fnref18" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn19"><a href="#fnref19" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn20"><a href="#fnref20" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn21"></li>
<li id="fn22"></li>
<li id="fn23">Table D.1 in <a href="#fnref23" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn24"></li>
<li id="fn25"><a href="#fnref25" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn26"><a href="#fnref26" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn27"></li>
<li id="fn28"></li>
<li id="fn29"><p>Forefront |url=<a
href="https://www.forefront.ai/blog-posts/gpt-j-6b-an-introduction-to-the-largest-open-sourced-gpt-model">https://www.forefront.ai/blog-posts/gpt-j-6b-an-introduction-to-the-largest-open-sourced-gpt-model</a>
|access-date=2023-02-28 |website=www.forefront.ai |language=en
|archive-date=2023-03-09 |archive-url=<a
href="https://web.archive.org/web/20230309205439/https://www.forefront.ai/blog-posts/gpt-j-6b-an-introduction-to-the-largest-open-sourced-gpt-model">https://web.archive.org/web/20230309205439/https://www.forefront.ai/blog-posts/gpt-j-6b-an-introduction-to-the-largest-open-sourced-gpt-model</a>
|url-status=dead }}<a href="#fnref29" class="footnote-back"
role="doc-backlink">↩︎</a></p></li>
<li id="fn30"></li>
<li id="fn31"><a href="#fnref31" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn32"><a href="#fnref32" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn33"></li>
<li id="fn34"></li>
<li id="fn35"><a href="#fnref35" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn36"><a href="#fnref36" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn37"><a href="#fnref37" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn38"></li>
<li id="fn39"><a href="#fnref39" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn40"></li>
<li id="fn41"></li>
<li id="fn42"></li>
<li id="fn43"><a href="#fnref43" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn44"></li>
<li id="fn45">Table 20 and page 66 of <em><a
href="https://storage.googleapis.com/pathways-language-model/PaLM-paper.pdf">PaLM:
Scaling Language Modeling with Pathways</a></em><a href="#fnref45"
class="footnote-back" role="doc-backlink">↩︎</a></li>
<li id="fn46"></li>
<li id="fn47"></li>
<li id="fn48"></li>
<li id="fn49"><a href="#fnref49" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn50"><a href="#fnref50" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn51"></li>
<li id="fn52"></li>
<li id="fn53"></li>
<li id="fn54"></li>
<li id="fn55"><a href="#fnref55" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn56"></li>
<li id="fn57"></li>
<li id="fn58"></li>
<li id="fn59"></li>
<li id="fn60"></li>
<li id="fn61"><a href="#fnref61" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn62"><a href="#fnref62" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn63"></li>
<li id="fn64"><a href="#fnref64" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn65"></li>
<li id="fn66"></li>
<li id="fn67"><a href="#fnref67" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn68"><a href="#fnref68" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn69"><a href="#fnref69" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn70"><a href="#fnref70" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn71"><a href="#fnref71" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn72"><a href="#fnref72" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn73"><a href="#fnref73" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn74"><p>AWS Machine Learning Blog |url=<a
href="https://aws.amazon.com/blogs/machine-learning/alexatm-20b-is-now-available-in-amazon-sagemaker-jumpstart/">https://aws.amazon.com/blogs/machine-learning/alexatm-20b-is-now-available-in-amazon-sagemaker-jumpstart/</a>
|website=aws.amazon.com |access-date=13 March 2023 |date=17 November
2022}}<a href="#fnref74" class="footnote-back"
role="doc-backlink">↩︎</a></p></li>
<li id="fn75"></li>
<li id="fn76"></li>
<li id="fn77"><a href="#fnref77" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn78"></li>
<li id="fn79"><a href="#fnref79" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn80"></li>
<li id="fn81"><a href="#fnref81" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn82"><a href="#fnref82" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn83"><a href="#fnref83" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn84"></li>
<li id="fn85"><a
href="https://www.businesswire.com/news/home/20230531005608/en/UAE&#39;s-Falcon-40B-World&#39;s-Top-Ranked-AI-Model-from-Technology-Innovation-Institute-is-Now-Royalty-Free">UAE's
Falcon 40B, World's Top-Ranked AI Model from Technology Innovation
Institute, is Now Royalty-Free</a>, 31 May 2023<a href="#fnref85"
class="footnote-back" role="doc-backlink">↩︎</a></li>
<li id="fn86"><a href="#fnref86" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn87"><a href="#fnref87" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn88"><a href="#fnref88" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn89"><a href="#fnref89" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn90"><a href="#fnref90" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn91"><a href="#fnref91" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn92"></li>
<li id="fn93"></li>
<li id="fn94"><a href="#fnref94" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn95"><a href="#fnref95" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn96"></li>
<li id="fn97"><a href="#fnref97" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn98"><a href="#fnref98" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn99"><a href="#fnref99" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn100"><a href="#fnref100" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn101"><a href="#fnref101" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn102"><a href="#fnref102" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn103"><a href="#fnref103" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn104"><a href="#fnref104" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn105"><a href="#fnref105" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn106"><a href="#fnref106" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn107"></li>
<li id="fn108"><a href="#fnref108" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn109"><a href="#fnref109" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn110"><a href="#fnref110" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn111"><a href="#fnref111" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn112"><a href="#fnref112" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn113"><a href="#fnref113" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn114"><a
href="https://x.com/karpathy/status/1781047292486914189">Andrej Karpathy
(Apr 18, 2024), <em>The model card has some more interesting info
too</em></a><a href="#fnref114" class="footnote-back"
role="doc-backlink">↩︎</a></li>
<li id="fn115"><a href="#fnref115" class="footnote-back"
role="doc-backlink">↩︎</a></li>
</ol>
</section>

## See also

-   [Foundation models](Foundation_models "wikilink")

## Notes

## References

## Further reading

-   [Jurafsky, Dan](Dan_Jurafsky "wikilink"), Martin, James. H. [*Speech
    and Language Processing: An Introduction to Natural Language
    Processing, Computational Linguistics, and Speech
    Recognition*](https://web.stanford.edu/~jurafsky/slp3/ed3book_jan72023.pdf),
    3rd Edition draft, 2023.

-   

-   

-   

-   [Open LLMs repository](https://github.com/eugeneyan/open-llms) on
    [GitHub](GitHub "wikilink").

-   

-   

[ ](Category:Large_language_models "wikilink") [Category:Deep
learning](Category:Deep_learning "wikilink") [Category:Natural language
processing](Category:Natural_language_processing "wikilink")

[1]

[2]

[3]

[4]

[5]

[6]

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

[20]

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

[47] Section 2.1 and Table 1,

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

[59]

[60]

[61]
An Open-Ended Embodied Agent with Large Language Models
|url=<https://voyager.minedojo.org/> |access-date=2023-06-09
|website=voyager.minedojo.org}}

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

[73]

[74]

[75]

[76]

[77]

[78]

[79]

[80]

[81]

[82]

[83]

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

[94] *[A Closer Look at Large Language Models Emergent
Abilities](https://www.notion.so/A-Closer-Look-at-Large-Language-Models-Emergent-Abilities-493876b55df5479d80686f68a1abd72f)*
(Yao Fu, Nov 20, 2022)

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

[108]

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

[137]

[138]
Citing Lee et al 2022.

[139]
.

[140]

[141]

[142]

[143]

[144]

[145]

[146]

[147]

[148]

[149]
