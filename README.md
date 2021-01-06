# Dater-to-Vec
Collaborative filtering in dating : a NLP-based user embedding approach inspired from Tinder's TinVec.

Table of content
- [An Overview of TinVec ](#overview)
- [Training setting](#setting)
- [Results](#results)
- [Usage](#usage)


## An Overview of TinVec
<a name="overview"/>

In this MLconf SF 2017 [presentation](https://www.youtube.com/watch?v=j2rfLFYYdfM) ([Sideshare](https://fr.slideshare.net/SessionsEvents/dr-steve-liu-chief-scientist-tinder-at-mlconf-sf-2017)), Dr. Steve Liu, Chief Scientist at Tinder, exposes an innovative approach to represent dating profiles and make recommendation based on users swiping behavior.

In short, the idea is that (subjectively) similar users will often be liked by raters with similar tastes: rated users hence ofen appear in a common context. If we could create vectorial representations of users that could express these common context, we would have a synthetic, profile-agnostic representation of users that would be useful to make profile recommendations.

Fortunaly, there is a class of NLP algorithms designed to do just that! Models like Word2Vec, given a corpus of sentences, learn to represent words that appear in similar verbal contexts. If we consider the ids of users that were co-liked by a specific rater as a sentence, Word2Vec will learn to give a close embeddings to users that are often liked simultaneously with the same other users.

This approach as the merit of relying on actual swiping behavior shown by users, which may capture subconscious patterns of choice. It does so without explicit consideration for the users profiles and is hence privacy-preserving by design.

How recommendations are then computed as well as limits in application are discussed below.

## Training setting
<a name="setting"/>

### Data
- We use the [dating collaborative filtering dataset from LibimSeTi](http://www.occamslab.com/petricek/data/). This dataset gives us access to realist dating behaviors, for a large number of users, in the form of ratings.

    > These files contain 17,359,346 anonymous ratings of 168,791 profiles made by 135,359 LibimSeTi users as dumped on April 4, 2006.

    > Ratings are on a 1-10 scale where 10 is best (integer ratings only)

    > Only users who provided at least 20 ratings were included

- To align with TinVec settings (i.e. Tinder swiping system) We transform ratings into likes/passes by defining a rating threshold for each user as the 85% (integer) quantile of their ratings. For instance: a rater who rated other users 7/10 and higher 17% of the time, and 8/10 and higher 10% of the time, is considered to have expressed interest in (i.e. liked) users they rated 7/10 and higher, which accounts for 17% of ratees.
- Data is split into train and test datasets with a stratification on raters id, holding out 10% of interactions for each rater.

### Embedding rated users - *Who likes you?*

Word2Vec learns to predict a word from its context (or the other way around in the skip-gram alternative to CBOW). Ideally we would generate all *(context,word)* combinations as training data, but this is intractable. Indeed, for a single rater with n=25 co-liked users, there are 365,650 different possible combinationsfor a W2V window of size 5. Therefore, we adopt the  strategy of a fast, random sampling of these combinations:
- Input data for each rater is a sentence of which the tokens are the ids of co-liked people: [rated_id_1, rated_id_2,...,rated_id_n]
- After each training epoch, all sentences are ramdomly shuffled so as to generate a new sampling of the combinations. To do so, and still use `gensim` Word2Vec class without modification, we use a custom iterator function.
- An additional benefit of this method is that the importance of each raters in training is only linearly proportionnal to their number of likes.

With a window of size 5, we then learn embeddings of size 200 over 100 trainng epochs, for all rated users with at least three likes.

### Embedding raters - *Who do you like?*

The raters taste is simply computed as the mean of the embeddings of the user they liked. This simple approach defines the "preferred type" of each rater i.e. an average representation of the kind of user they like.

### Predicting affinity between two users - *Are they your type?*

Euclidian distance between raters and ratees embeddings is a straightforward expression of how dissimilar a ratees might be from people previously liked by the rater: the smaller the distance, the more likely the like!

To predict matches, we train a classifier on a subset of the training data (200,000 records) as we are dealing with simple 1D inputs. We evaluate on the full test data (1,735,935 records). We use the LGBM classifier, which is way too complex for this task but was already set up.

Limits in application: recommendations based on a "preferred type" has obvious conceptual limits, with the risk of a self-reinforcing feeedback loop as well as a poorer, less diverse user experience.
Using raw embeddings as training data could be an alternative in order to learn more complex preferences than simple distance. Morever, occasionnaly recommending users different from the preferred type might be a way to keep a diverse, serendipitous experience (clustering might be an interesting approach to do so).



# Usage
<a name="usage"/>


Switch between train and developping mode by modifying `DEV_MODE` in `src/config.py`.

Get the data and split into test and training, optionnaly up to a certain rater ID with:

    python src/processing.py [--max_id 150000

Train the Word2Vec model (step 1), learn raters embeddings (step 2) and get the rater-rated distance (step 3) with:

    python train.py [--steps 123] [--resume_training n]


# Results
<a name="overview"/>

[WIP]

Link between euclidian distances between embeddings and matches in train set (1,735,935 records)

Spearman correlation :

- A plot of vector difference norm and proportion of people that matched or not.

- Design and performance using vector difference and LightGBM.

- Visualization of embedded users and wether they were matched or not by a specific user
