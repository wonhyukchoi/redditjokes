# Deduplicates and returns number of reposts in /r/jokes.
#
# Future additions may include deduplication of any subreddit of choice.
# Currently, using this class not as the main class is not recommended.
#
# Required libraries: numpy, pandas, SetSimilaritySearch, nltk 
# Note: After downloading `nltk`, you will need to open a `python` console and run:
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# otherwise, you can comment out the preprocessing and otherwise load the pickle.

#Thanks to Eric Zhu @ https://github.com/ekzhu/SetSimilaritySearch
import numpy as np
import pandas as pd
import string, time
from tqdm import tqdm
from SetSimilaritySearch import all_pairs
from collections import Counter, defaultdict
from wrappers import timer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Combines two text columns of a dataframe and adds it to a new column.
# ARGS: dataframe, and columns to be added
# Returns: a dataframe with the new column
def combine_df_cols(df, col1 = 'title', col2 = 'content'):
    text = []
    for i,j in df[[col1,col2]].values:
        text.append(str(i) + " " + str(j))
    df['text'] = text
    return df


# Unused function, but use if you want to get frequencies directly from 
# get_similar_results without passing through get_sets. 
# ARGS: a mxn list
# Returns: An descending order sorted numpy array by frequency of the first and second columns.
def freq_counts(result):
    
    # Unrolls the first and second columns into a 1d list.
    def unroll(result):
        array = np.array(result)
        return [int(i) for i in array[:,:2].reshape(-1,).tolist()]
    
    # Counts the number of values in the 1d list and sorts it
    counts = Counter(unroll(result))
    unsorted = np.array([[k,v] for k,v in counts.items()])
    return unsorted[unsorted[:,1].argsort()[::-1]]

# Returns pairs with similarity above certain threshold
# ARGS: a list of list of strings(txts split into words),
#      the similarity function(jaccard or cosine), the threshold to show results
# Returns: a nx3 list of showing similar texts . col1, col2 = similar texts, col3 = similarity.
def get_similar_results(txtlist, sim = 'jaccard', threshold = 0.4):
    pairs = all_pairs(txtlist, similarity_func_name=sim, 
            similarity_threshold=threshold)
    return list(pairs)    


# Returns a set given a list of words.
# ARGS: a list of words  
# outputs: a set of n-grams
def ngram(txtlist, n=3):
    output = set()
    # Get n-gram at each possible position
    for i in range(len(txtlist)+(1-n)):
        gram = []
        for j in range(n):
            gram.append(txtlist[i+j])
        output.add(tuple(gram))
    return output

# Text preprocessing by removing stopwords & non-ASCII chars, and applies wordnet lemmatizaiton.
# ARGS: txt,a string
# Ouputs: a pre-processed list of words
def preprocess_text(txt):
    lemmatizer = WordNetLemmatizer()
    
    word_list = simple_strip(txt)
    
    # Remove stopwords and lemmatize
    word_list = [word for word in word_list if word not in stopwords.words('english')]
    word_list = [lemmatizer.lemmatize(i) for i in word_list]
    word_list = [word for word in word_list if len(word) > 1] # remove 1-letter words
    
    return word_list

# Rudimentary preprocessing by only removing non-ASCII characters.
# ARGS: a string
# outputs: a list of words without \n and punctuation 
def simple_strip(txt):
    regex = RegexpTokenizer(r'\w+')
    
    # Returns only ASCII characters and strips punctuation
    txt = "".join([char for char in txt if char not in string.punctuation and ord(char)<128])
    word_list = regex.tokenize(txt.lower()) # naive splitting method, virtually equivalent to split() 
    
    return word_list

# Gets the set of reposts by using graph traversal to get connected components
# Source: https://stackoverflow.com/a/42036326
# ARGS: The similarity results(a list of tuples)
# Outputs: defaultdict of sets (the keys can be ignored)
@timer 
def get_sets(sim_results):
    def dfs(adj_list, visited, vertex, result, key):
        visited.add(vertex)
        result[key].append(vertex)
        for neighbor in adj_list[vertex]:
            if neighbor not in visited:
                dfs(adj_list, visited, neighbor, result, key)
    
    # Convert similarity results into a nx2 array
    edges = np.array(sim_results, dtype=int)[:,:2]

    # Convert similarity results to adjancency list
    adj_list = defaultdict(list)
    for x, y in edges:
        adj_list[x].append(y)
        adj_list[y].append(x)
    
    # Find connected components 
    result = defaultdict(list)
    visited = set()
    for vertex in adj_list:
        if vertex not in visited:
            dfs(adj_list, visited, vertex, result, vertex)

    return result

# ARGS: dataframe where the items come from, a list of sets, an the criterion to choose the set representative
# Outputs: post with highest number of upvotes(default), as well as its frequencies
@timer
def get_rep(df,result_sets, crit = 'upvotes'):
    output = []
    # Iterate through all sets
    for _,result_set in result_sets.items():

    	frequency = len(result_set)
    	if frequency == 0: continue # guard against empty sets

        max_score = -1
        chosen_one = None
        
        # Choose the item with the highest score 
        for item in result_set:
            score = int(df.iloc[item][crit])
            if score > max_score:
                max_score = score
                chosen_one = item
                
        output.append([chosen_one, frequency ])
    return np.array(output)

# ARGS: The dataframe and the column to deduplicate, preprocess params, set similiarity serach params
# Outputs:Frequency as well as other items of the representative repost
def repost_results(df_in, column='text', preprocess = 'normal', sim= 'jaccard', threshold = 0.4):
    # Combine title and content
    df_in = combine_df_cols(df_in)    
    
    # Preprocessing takes the most time.
    # Comment this part if you want to use the pickle.
    if preprocess == 'simple':
        txt = [ngram(simple_strip(item)) for item in tqdm(df_in[column])]
    elif preprocess == 'normal':
        txt = [ngram(preprocess_text(item)) for item in tqdm(df_in[column])]
    else: 
        raise ValueError('Unaccepted type of preprocessing')

    """ Since preprocessing takes time, you can use the preprocessed pickle.
        Comment the preprocessing steps above if you want to use this. 
    import pickle
    with open('2019joke_preprocessed_pickle','rb') as f:
        txt = pickle.load(f)
    """

    sim_results = get_similar_results(txt, sim, threshold) #All-pair similarity
    sets = get_sets(sim_results) # Get bucket of reposts
    rep = get_rep(df_in, sets) # Get representative from each bucket
    indices = rep[:,0] # Indices of the representative elements
    frequencies = rep[:,1] # Length of each repost set = frequency of reposts
    df = df_in.iloc[indices] 
    df['reposts'] = frequencies
    
    return df.sort_values(by='frequencies', ascending=False)


if __name__ == "__main__":
	df = pd.read_csv('2019_reddit_jokes.csv')
	reposts = repost_results(df)
	reposts.to_csv('2019_jokes_reposts.csv')

