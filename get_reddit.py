# Obtains reddit data from pushshift API.
#
# Required libraries: praw, psaw, pandas, tqdm 
#
# Future version may encapsulate the functions into a class 
# so as to provide reproducability. 

import praw
import pandas as pd
from psaw import PushshiftAPI
from tqdm import tqdm

# Unused function
# Just a reference function for using standard reddit API.
def jokes_top(reddit):
	df = pd.DataFrame()
	jokes = reddit.subreddit('jokes')
	for item in jokes.top(limit=None):
	    row = {'title':item.title, 'content':item.selftext,'upvotes':item.score, 
	          'url': item.url, 'time':item.created_utc}
	    df = df.append(row, ignore_index=True)
	return df

# Generates n x 2 list of timestamps offset by one day
def datelist_generator(start_date, end_date):
    dates = [int(day.timestamp()) for day in pd.date_range(start=start_date, end=end_date).tolist()]
    datelist = [[dates[n],dates[n+1]] for n in range(len(dates)-1)]
    return datelist

# Please, do not post edits in the joke subreddit.
# Edit: Mom! I've reached the front page!
def rm_edit(txt):
	txt = txt.lower()
	edit_index = txt.find('edit')
	if edit_index == -1: return txt
	else: return txt[:edit_index]

# Appends given results to dataframe
def append_results(df, results):
    for item in results:
        if item.selftext == '[deleted]' or item.selftext == '[removed]': continue

        row = {'title':item.title, 'content':rm_edit(item.selftext),'upvotes':int(item.score), 
              'url': item.url, 'comments': int(item.num_comments), 'time': item.created_utc}
        df = df.append(row, ignore_index=True)
    return df

# Append results given a datelist to a csv
# If you want to be "safe," modify to use csv.writer instead of df.to_csv
def flush_results(api, subreddit, start_date, end_date, file_name, limit=2000):

	datelist = datelist_generator(start_date, end_date)
	df = pd.DataFrame(columns={'title','content','upvotes','url','comments','time'})

	for dates in tqdm(datelist):
	    try:
		    gen = api.search_submissions(subreddit=subreddit,limit=limit,
		                                 after = dates[0], before = dates[1])
		    results = list(gen)
		    df = append_results(df, results)
	    except:
		    print('Getting data from {} failed.'.format(dates[0]))
		    pass

	df.to_csv(file_name)	


if __name__ == '__main__':
	reddit = praw.Reddit(user_agent = 'NAME OF YOUR USER AGENT HERE', client_id = 'YOUR ID HERE',
	                    client_secret = 'YOUR SECRET HERE')
	api = PushshiftAPI(reddit)

	start_date = '2019-01-01'
	end_date = '2019-12-31'

	flush_results(api, 'jokes', start_date, end_date, '2019_reddit_jokes.csv')



