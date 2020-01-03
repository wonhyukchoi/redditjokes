# Reddit /r/jokes reposts by machine learning

The r/jokes subreddit is famous (infamous?) for its amount of reposts. 
This project aims to uncover the amount of reposts, and find out which jokes are reposted more frequently.

#### Required libraries:

Install with `pip install -r requirements.txt`. 

Note: After downloading `nltk`, you will need to open a `python` console and run:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Methodology
### Data collection

I obtained all reddit jokes post from the year 2019 using the [Python reddit API wrapper(PRAW)](https://praw.readthedocs.io/en/latest/) and the [Python Pushshift.io API Wrapper (PSAW)](https://github.com/dmarx/psaw) to obtain the data. I collect all posts by a daily interval. Generally, there are around about [500~600 posts in /jokes daily](http://api.pushshift.io/reddit/submission/search/?subreddit=jokes&aggs=created_utc&frequency=day&after=30d&size=0). 

I deleted all posts whose content are "[deleted]" or "[removed]". I'm very well aware I'm missing out on great jokes that have these as their body, but 

### Data analysis

The title and the contents of the posts were combined into one text. The pre-processing steps are remarkably standard (non-ASCII removal, stopword removal, tokenization, removal of words shorter than 2 chars). Each text is then processed into a set of its 3-grams. 

In order to find the reposts, I utilized [Eric Zhu's SetSimilaritySearch library](https://github.com/ekzhu/SetSimilaritySearch), which returns a fast all-pairs similarity search. Then I find all "repost groups" by a standard graph traversal algorithm (dfs), in order to find the number of reposts per joke and find the post with the largest amount of upvotes in the repost group.

Note: Most reposts are word-for-word copies, and it is unnecessary to semantically embed(i.e. word2vec) the phrases. Typically computing cosine or jaccard similarities between all vectors can be incredibly costly (a brute-force algorithm gives $O(n^2)$ time) but the [all-pair binary algorithm](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/32781.pdf) gives a very fast implementation. Otherwise, locality sensitive hashing (LSH) is another popular algorithm that forces hash collisions on similar items and thereby clustering them.

## Results

### Data

Overall, I obtained 

This includes XXX posts that have a combined amount of XXX reposts. 

For the top reposts, refer back above. 

### Parameters

I found that a jaccard similarity of 0.4 on 3-grams generally guarantees that all pairs _are_ reposts. It will work sufficiently well with a similarity of 0.3, but it will occasionally misclassify different jokes as reposts. I did not build a test set to evaluate my similarity values, and I have used a more conservative threshold of 0.4.

Jaccard v.s. cosine similarity, different similarity thresholds, different n-grams, different preprocessing may all result in an improved repost detection. 