
# coding: utf-8

# #Data cleaning

# Data cleaning steps:
# - replace @ handles with: hdl (chose a word with no punctuation so no extra step is needed to process, hdl = handle)
# - replace urls by: url
# - replace emoticons with corresponding emojis
# - split retweets in tweet and retweet
# 
# For all of these, first check what is the prevalence and if it is worth the effort. 
# 
# Ideas:
# - split hashtags into words -> Carlo tackling that
# - Tweets not in english -> Daniel tackling that
# - replace contractions & spellchecking
#     - Character ngram will probably be more efficient due to the really low quality of speach

# In[4]:

import json
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
pd.options.display.max_colwidth = 140
import nltk
import re
from IPython.display import display
import src.happyfuntokenizing
# from src.happyfuntokenizing import TweetTokenizer

from math import log


# In[ ]:

src.happyfuntokenizing.Tokenizer


# # Key functions

# In[2]:

# placeholder cannot be called in this file before the all subfunctions are defined
def clean(df):
    """Data cleaning steps:
    - replace @ handles with: hdl (chose a word with no punctuation so no extra step is needed to process, hdl = handle)
    - replace urls by: url
    - replace emoticons with corresponding emojis
    - split retweets in tweet and retweet
    - remove non english tweets, done on the text only (not retweet). Needs to be applied after emoji split. 
    - hashtags splits
    - subset by only necessary fields
    
    Returns None (data cleaning in place)
    """
    cleanHandle(df)
    cleanURL(df)
    convertEmoticon(df)
    cleanRetweets(df)
    splitTextEmoji(df)
    df = cleanNonEnglish(df)
    
    tokenize_total_text(df)
    total_text = tokenize_total_text(df)
    """ wordcost and maxword used from 'Generic Human' on stackoverflow question
        http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words"""
    wordcost = dict((k, log((i+1)*log(len(total_text)))) for i,k in enumerate(total_text))
    maxword = max(len(x) for x in total_text)
    
    run_hashtag(df, maxword, wordcost)
    df = df.drop(['lat','lng','only_text','timeStamp'], axis=1)
    
    return df
    


# In[3]:

def loader(filename):
    """ Load tweets from filename. Resets the index. Returns the loaded data frame"""
    with open(filename,'r') as f:
        df = pd.DataFrame(json.load(f))
    df.reset_index(inplace=True, drop=True)
    return df


# # Support functions

# ### Replace @ handles with hdl

# In[4]:

def cleanHandle(df):
    """ Replace in-place handles with hdl keyword
    
    Returns None
    """
    pattern = r"@[a-zA-Z0-9_]{1,15}" #from http://kagan.mactane.org/blog/2009/09/22/what-characters-are-allowed-in-twitter-usernames/
    print("{} handles replaced".format(np.sum(df.text.str.contains(pattern).values)))
    df.text = df.text.str.replace(pattern, " hdl ")
    return


# No "hdl" words for confusion in the txt. Good replacement name. 
# 
# Best to replace handles with "\ hdl\ ", so for tokenization it will be easier to identify as a word. 

# ### Replace URLs with url

# In[5]:

def cleanURL(df):
    """ Replace in-place URLs with url keyword
    
    Returns None
    """
    pattern = r'(?:http://|https://|www.)[^“”"\' ]+' # From http://stackoverflow.com/questions/7679970/python-regular-expression-nltk-website-extraction
    print("{} urls replaced".format(np.sum(df.text.str.contains(pattern).values)))
    df.text = df.text.str.replace(pattern, " url ")
    return


# URLs are quiet well formed and are generally at the end of tweets. No risk of engulfing in the cleaning some more text after the url.
# 
# keyword url is used only 4 times in dataset, no risk of confusion

# ### Convert emoticons to emojis

# In[6]:

# Based on:
# https://slack.zendesk.com/hc/en-us/articles/202931348-Emoji-and-emoticons
# http://unicodey.com/emoji-data/table.htm
# http://www.unicode.org/emoji/charts/emoji-list.html

def convertEmoticon(df):
    """ Replace in-place common emoticons to emojis.
    
    Returns None
    """
    emoticon2emoji = {
        r"<3": "\u2764",
        r"</3": "\U0001F494",
        r"8\)": "\U0001F60E",
        r"D:": "\U0001F627",
        r":'\(": "\U0001F622",
        r":o\)": "\U0001F435",
        r":-*\*": "\U0001F48B",
        r"=-*\)": "\U0001F600",
        r":-*D": "\U0001F600",
        r";-*\)": "\U0001F609",
        r":-*>": "\U0001F606",
        r":-*\|": "\U0001F610",
        r":-*[Oo]": "\U0001F62E",
        r">:-*\(": "\U0001F620",
        r":-*\)|\(:": "\U0001F603",
        r":-*\(|\):": "\U0001F61E",
        r":-*[/\\]": "\U0001F615",
        r":-*[PpbB]": "\U0001F61B",
        r";-*[PpbB]": "\U0001F61C"
    }
    
    total = 0
    for emoticon in emoticon2emoji:
        nreplacements = np.sum(df.text.str.contains(emoticon).values)
        total += nreplacements
        print("{:10} -> {:>5} replaced {:6} times".format(emoticon, emoticon2emoji[emoticon], nreplacements))
        df.text = df.text.str.replace(emoticon, emoticon2emoji[emoticon])
    print("{:3} replaced {} times".format("ALL", total))
    return


# ### Split retweets into user content and retweeted content

# In[7]:

# Cannot be called before functions within are defined 
def cleanRetweets(df):
    """ Remove from the text column the retweet column and add to seperate column "retweet". 
    
    Returns None
    """
    nsplits = 0
    nsplits += splitRetweets(df)
    nsplits += splitQuotes(df) # Need to maintain that order
    print("{} retweets processed".format(nsplits))
    return


# In[8]:

def splitRetweets(df):
    """ Extract retweets with the RT keyword"""
    pattern = r"""(.*(?:\W|^))(RT(?:\ ?[\":“]|$).*) # Retweets with RT keyword"""
    retweets = pd.DataFrame(df.text.str.extract(pattern, flags=re.X))
    retweets.columns = ["text", "retweet"]
    non_null_idxs = retweets.retweet.notnull()
    df.loc[non_null_idxs,["text"]] = retweets.text[non_null_idxs]
    df["retweet"] = retweets.retweet
    return len(non_null_idxs)


# In[9]:

def splitQuotes(df):
    """ Extract retweets in quote format.
    See http://support.gnip.com/articles/identifying-and-understanding-retweets.html
    """
    pattern = r"(.*?)((?:([\"\'])|(?:(“))|‘)\s*hdl.*(?(3)\3|(?(4)”|’)))(.*)" #Pattern to match quote, possibly nested
    retweets = pd.DataFrame(df.text.str.extract(pattern, flags=re.X)[[0,1,4]])
    retweets.columns = ["text_before", "retweet", "text_after"]
    non_null_idxs = retweets.retweet.notnull()
    retweets["text"] = retweets.loc[non_null_idxs, ["text_before", "text_after"]].apply(lambda x: " ".join(x), axis=1)
    df.loc[non_null_idxs,["text"]] = retweets.text[non_null_idxs].copy()
    df["retweet"] = retweets.retweet.copy()
    return len(retweets[non_null_idxs])


# ### Split Text and Emoji and create two new columns for only text and only emoji

# In[10]:

try:
    # Wide UCS-4 build
    highpoints = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+', 
        re.UNICODE)
except re.error:
    # Narrow UCS-2 build
    highpoints = re.compile(u'('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF])+', 
        re.UNICODE)
# Functions to check whether there's an emoji in the text, return 1 if true, 0 if false
def is_emoji(text):
    if highpoints.search(text):
        return 1
    else:
        return 0
def splitTextEmoji(df):
    tok = happyfuntokenizing.TweetTokenizer(preserve_case=False)
    def emojiExtract(sent):
        return [word for word in tok.tokenize(sent) if is_emoji(word) == 1]

    def textExtract(sent):
        return ''.join([word for word in sent if is_emoji(word) == 0])

    def addEmoji(df):
        df['only_emoji'] = [emojiExtract(word) for word in df.text]

    def addText(df):
        df['only_text'] = [textExtract(word) for word in df.text]
    
    addText(df)
    addEmoji(df)
    return


# ## Functions to clean non-english columns

# In[11]:

import string
punctuation = string.punctuation
ex = ['“', '—', '’', ' ️', '️', '...', '”', '…', ' , , ,', '?', ' ', ' ⃣', '∞', '🆒']
for pun in [word for word in ex if word not in punctuation]:
    punctuation += pun
    
def isEnglish(list):
    try:
        [word.encode('ascii') for word in list if word not in punctuation]
    except Exception:
        return False
    else:
        return True
def cleanNonEnglish(df):
    """ 
    
    Needs to be applied after emoji splitting as emojis are considered non-english
    """
    text_list = df['only_text'].values
    english_Boolean = [isEnglish(sent) for sent in text_list]
    print("{} number of tweets are not English".format(len(english_Boolean) - sum(english_Boolean)))
    return df[english_Boolean]


# # Hashtag Split

# In[12]:

def tokenize_total_text(df):
    tokenizer = TweetTokenizer()
    total_text = df.text.apply(lambda x: tokenizer.tokenize(x))
    
    stop_words = nltk.corpus.stopwords.words('english') + ["http", 'hdl', 'url'] 
    punctuation_words = list(set(string.punctuation)) + [":", ":/"]

    total_text = list(total_text)
    total_text = [word for _list in total_text for word in _list if word not in stop_words and word not in punctuation_words]
    
    return total_text


# In[13]:

def run_hashtag(df, maxword, wordcost):
    def addHashTag(df):
        df['only_HashTag'] = [re.findall(r"(#\w+)", word) for word in df.text]

    def is_hashtag(text):
        if re.search(r"(#\w+)", text):
            return 1
        else:
            return 0

    def textEmojiExtract(sent):
        tok = happyfuntokenizing.TweetTokenizer(preserve_case=False)
        return ' '.join([word for word in tok.tokenize(sent) if is_hashtag(word) == 0])

    def textHashtagSplit(sent):
        tok = happyfuntokenizing.TweetTokenizer(preserve_case=True)
        return ' '.join([word if is_hashtag(word) == 0 else apply_hashtag_text(word, maxword, wordcost) for word in tok.tokenize(sent)])

    def noHashtagText(df, wordcost):
        df['only_text_splithashtag'] = [textHashtagSplit(word) for word in df.only_text]
    #     df['text_splithashtag'] = [textHashtagSplit(word) for word in df.text]
        return

    def infer_spaces(s, maxword, wordcost):
        """ Code used from 'Generic Human' on stackoverflow question
        http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words"""

        # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
            return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        return list(reversed(out))

    def apply_hashtag_split(_hashtag, maxword, wordcost):
        # for each hashtag in list of hashtags, split on # and take second item
        item = [item.split('#')[1] for item in _hashtag if len(item) > 0]

        # regex pattern
        # (1) split on two caps or more, and only keep caps
        # (2) split on exactly one cap or more, and keep trailing letters
        pattern = r'([A-Z]{2,})|([A-Z]{1}[a-z]*)'

        # loop through each item in list of hashtags
        final_hashtags = []
        for word in item:

            # if len of word is 0, then there is no hashtag
            if len(word) == 0:
                final_hashtags.append("empty_hashtag")

            # use (1)) regex: funciton lowercase, as "Treatlowercase" can be treated as lowercase
            elif word[0].isupper() and word[1:].islower():
                final_hashtags.append(infer_spaces(word.lower(), maxword, wordcost))

            # use (1) regex: funciton lowercase
            elif word.islower():
                final_hashtags.append(infer_spaces(word, maxword, wordcost))

            # use (2) regex: customized uppercase
            else:
                final_hashtags.append(list(filter(None, re.split(pattern, word))))
        return final_hashtags

    def apply_hashtag_text(_hashtag, maxword, wordcost):
        word = _hashtag.split('#')[1] 

        # regex pattern
        # (1) split on two caps or more, and only keep caps
        # (2) split on exactly one cap or more, and keep trailing letters
        pattern = r'([A-Z]{2,})|([A-Z]{1}[a-z]*)'

        # if len of word is 0, then there is no hashtag
        if len(word) == 0:
            final_hashtags = ""

        # use (1)) regex: funciton lowercase, as "Treatlowercase" can be treated as lowercase
        elif word[0].isupper() and word[1:].islower():
            final_hashtags =  infer_spaces(word.lower(), maxword, wordcost)

        # use (1) regex: funciton lowercase
        elif word.islower():
            final_hashtags = infer_spaces(word, maxword, wordcost)

        # use (2) regex: customized uppercase
        else:
            final_hashtags = list(filter(None, re.split(pattern, word)))

        joined_hashtags = ' '.join(final_hashtags)
        return joined_hashtags
    
    addHashTag(df)
    noHashtagText(df, wordcost)
    df["split_hashtag"] = df.only_HashTag.apply(apply_hashtag_split, maxword=maxword, wordcost=wordcost)


# # Save clean file

# In[14]:

if __name__ == "__main__":

    clean_test_tweets_df = loader('./data/tweets_test.json')
    clean_test_tweets_df = clean(clean_test_tweets_df)
    clean_test_tweets_df.to_json('./data/tweets_test_clean.json', force_ascii=False)
    
    clean_training_tweets_df = loader('./data/tweets_training.json')
    clean_training_tweets_df = clean(clean_training_tweets_df)
    clean_training_tweets_df.to_json('./data/tweets_training_clean.json', force_ascii=False)


# In[ ]:



