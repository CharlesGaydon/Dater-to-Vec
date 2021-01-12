# Dater-to-Vec
Collaborative filtering in dating : a NLP-based user embedding approach inspired from Tinder's TinVec.
___
- [Dater-to-Vec](#dater-to-vec)
  - [An Overview of TinVec](#an-overview-of-tinvec)
  - [Training setting](#training-setting)
    - [Data](#data)
    - [Embedding rated users - *Who likes you?*](#embedding-rated-users---who-likes-you)
    - [Embedding raters - *Who do you like?*](#embedding-raters---who-do-you-like)
    - [Predicting affinity between two users - *Are they your type?*](#predicting-affinity-between-two-users---are-they-your-type)
- [Results](#results)
- [Code Usage](#code-usage)
- [Appendix](#appendix)
___


## An Overview of TinVec

In this MLconf SF 2017 [presentation](https://www.youtube.com/watch?v=j2rfLFYYdfM) ([Sideshare](https://fr.slideshare.net/SessionsEvents/dr-steve-liu-chief-scientist-tinder-at-mlconf-sf-2017)), Dr. Steve Liu, Chief Scientist at Tinder, exposes an innovative approach to represent dating profiles and make recommendation based on users swiping behavior.

In short, the idea is that (subjectively) similar users will often be liked by raters with similar tastes: rated users hence ofen appear in a common context. If we could create vectorial representations of users that could express these common context, we would have a synthetic, profile-agnostic representation of users that would be useful to make profile recommendations.

Fortunaly, there is a class of NLP algorithms designed to do just that! Models like Word2Vec, given a corpus of sentences, learn to represent words that appear in similar verbal contexts. If we consider the ids of users that were co-liked by a specific rater as a sentence, Word2Vec will learn to give a close embeddings to users that are often liked simultaneously with the same other users.

This approach as the merit of relying on actual swiping behavior shown by users, which may capture subconscious patterns of choice. It does so without explicit consideration for the users profiles and is hence privacy-preserving by design.

> By experimenting with this approach on a custom classification task based on real dating data, we were able to predict with high confidence if dating app users will "like" other users, with a **ROC AUC of 87% and an accuracy of 86%**

Below, we discuss data, methods and results.

## Training setting

### Data

- We use the [dating collaborative filtering dataset from LibimSeTi](http://www.occamslab.com/petricek/data/). This dataset gives us access to realist digital dating behaviors, for a large number of users, in the form of 1-10 ratings.

    > These files contain 17,359,346 anonymous ratings of 168,791 profiles made by 135,359 LibimSeTi users as dumped on April 4, 2006.

    > Ratings are on a 1-10 scale where 10 is best (integer ratings only)

    > Only users who provided at least 20 ratings were included

- Data is split into train and test datasets with a stratification on raters id, holding out 10% of interactions for each rater.
- To align with TinVec settings (i.e. approcimate the Tinder swiping system), we transform ratings into likes/passes by defining a rating threshold for each user as the 85% (integer) quantile of their ratings, only considering train data. For instance: a rater who rated other users 7/10 and higher 17% of the time, and 8/10 and higher 10% of the time, is considered to have expressed interest in (i.e. liked) users they rated 7/10 and higher, which accounts for 17% of ratees. This resulted in ~37% of positive class.


### Embedding rated users - *Who likes you?*

The skip-gram approach of Word2Vec with negative sampling learns to predict if a words belongs to the context of a target word. Ideally we would generate all *(context,target)* combinations as training data, but this is intractable: for a single rater who liked n=25 other users, there are 365,650 different possible _(context, target)_ combinations if we use a context window of size 5.

Therefore, we design a strategy of fast, random sampling of these combinations:
- Input data for each rater is a sentence of which the tokens are the ids of other users that they liked: [rated_id_1, rated_id_2,...,rated_id_n]
- After each pass of the model on the data, all sentences are ramdomly shuffled so as to generate a new sampling of the combinations. To do so, while still using `gensim` Word2Vec class without modification, we use a custom iterator function.
- To have a slow decrease of the alpha learning rate (which is controled by the gensim Word2Vec class), we count 5 passes on the data for each epoch.
- An additional benefit of this method is that the importance of each raters in training is only linearly proportionnal to their number of likes.

We use the skip-gram algorithm, with a window of size 5, and learn embeddings of size 200 over 50 trainng epochs, for all rated users with at least 3 likes. We use negative sampling (n=5).


### Embedding raters - *Who do you like?*

The raters taste is simply computed as the mean of the embeddings of the user they liked. This simple approach defines the "preferred type" of each rater i.e. an average representation of the kind of user they like.


### Predicting affinity between two users - *Are they your type?*

TinVec approach uses the straightforward Euclidian distance between raters and ratees embeddings as an expression of how dissimilar a ratees might be from people previously liked by the rater: the smaller the distance, the more probable the like is thought to be. There are obvious limits: recommendations based on a "preferred type" bears the risk of a self-reinforcing feeedback loop as well as a poorer, less diverse user experience. On a side note, occasionnaly recommending users different from the preferred type might be a way to keep a diverse, serendipitous experience (clustering might be an interesting approach to do so).

Following this distance-based approach, we trained a classifier to find the right threshold. We use a LightGBM, which is oversized for the task but was already ready set up from previous experiments. We consider only a subset of the training data (200,000 records) as we are dealing with simple 1D inputs. **This approach lead to poor results (see below).** Furthermore, preprocessing the data by computing the distance was computationnaly intensive, and the evaluation of embeddings on the classification task would take a long time (at least one hour).

Alternatively, we tried using the pre-trained embeddings *via* embeddings layers in a neural network, which permits to learn more complex preferences patterns than simple euclidian distance, without the tedious data preprocessing.
In practice, a Keras neural net was created, with two separate, unmutable embeddings, initialized with zeros and then filled with the pretrained embeddings when possible. Their concatenation is followed by three dense layers: Dense(50, relu), Dense(25, relu) and the final Dense(50, sigmoid). Adam optimizer is used with cross_entropy loss. A random 10% fraction of the train data is kept as validation data, for early stopping based on the loss, with no patience.
**This gave very good results (see below).**

Making the embeddings layer trainable could lead to even better results but was slow: an epoch was estimated to last >16 hours, which made the comparison inpractical. We therefore do not compare the two approaches (trainable vs. non-trainable embeddings).


# Results

✅ Word Embeddings: Word2Vec training took 8min24sec, with a trainng loss going from 22,440k to 775k (/30 division) over 50 epochs, plateauing after 40 epochs. This demonstrates that there is room to learn users context in such data, as proposed by the TinVec approach.

❌ Distance-based classification: euclidian distance between raters and ratees passed to LGBM lead to a 0.540 ROC AUC, only slightly better than a random classifier. What's more, (rater, ratee) pairs which did not match had on average slightly smaller distance that those who did. This goes against the TinVec idea that the distance between embeddings is in itself enough to predict matches.

✅ Embedding based classifification:
  - Training stopped as early as the 5th epoch, at which point validation loss started decreasing. The model reached a training and validation ROC AUC of around 0.95.

  ![Training ROC AUC](/results/training_roc_auc.png)
  - An evaluation on the test set demonstrated lower but still really good prediction performance, with a **ROC AUC of 87% and an accuracy of 86%**.

  ![Test ROC AUC ](/results/test_roc_auc.png)

Overall, our project demonstrated the potential of a collaborative filtering approach that learns to associate users to their "dating context" i.e. who they are co-liked with. We did not compare performances of this approach to learning embeddings from scratch during classifier training, but at least proved that it significantly (>20 times) speeds up the learning process.


# Code Usage

This code was run inside of an Anaconda Continuum 3 docker container, in a conda environment based on `d2v_env.yml`. A `requirements.txt` file specifies the packages versions for full reproductibility.

Switch between train and developping mode by modifying `DEV_MODE` in `src/config.py`.

Get the data and split into test and training, optionnaly up to a certain rater ID with:

    python src/processing.py [--max_id 150000]

Train the Word2Vec model (step 1), learn raters embeddings (step 2) and get the rater-rated distance (step 3) with:

    python train.py [--steps 123] [--resume_training n]

Train the final classifer by running the Jupyter notebook `notebooks/Embedding Classifier - trainable=False.ipynb` and show evaluation in `notebooks/Embedding Classifier - comparison.ipynb`.

Pre-commit can be used to apply `flake8` and `black` controls (and corrections). Run at root with:

    pre-commit

# Appendix

Keras embedding-based model summary:

    __________________________________________________________________________________________________
    Layer (type) Output Shape Param
    # Connected to
    ==================================================================================================
    input_13 (InputLayer) [(None, 1)] 0

     __________________________________________________________________________________________________
    input_14 (InputLayer) [(None, 1)] 0

     __________________________________________________________________________________________________
    embedding_12 (Embedding) (None, 1, 100) 13536000 input_13[0][0]

     __________________________________________________________________________________________________
    embedding_13 (Embedding) (None, 1, 100) 22097100 input_14[0][0]

     __________________________________________________________________________________________________
    concatenate_6 (Concatenate) (None, 1, 200) 0 embedding_12[0][0] embedding_13[0][0]

     __________________________________________________________________________________________________
    dense_18 (Dense) (None, 1, 50) 10050 concatenate_6[0][0]

     __________________________________________________________________________________________________
    dense_19 (Dense) (None, 1, 25) 1275 dense_18[0][0]

     __________________________________________________________________________________________________
    dense_20 (Dense) (None, 1, 1) 26 dense_19[0][0]

     ==================================================================================================

    Total params: 35,644,451 Trainable params: 11,351 Non-trainable params: 35,633,100

     __________________________________________________________________________________________________
