
# coding: utf-8

# # Word2Vec with emojis demo
# The goal of this demo is to experience the power of word2vec which works well also with emojis. 

# In[1]:

from data_cleaning import loader
from gensim.models import Word2Vec


# ### Load word2vec model
# For training see word2vec_hierarchical_clustering. The model is trained on 800k tweets cleaned, with a window size of 10 (for more semantic similarities), a feature vector of size 200 and otherwise standard parameters. 

# In[2]:

emoji_model = Word2Vec.load('emoji.embedding')


# ### Load emojis

# In[3]:

def convertEmojis(df):
    """Converts emojis df to printable format """
    emojis = list(map(lambda x: bytes("{}{}".format(*x), 'ascii').decode('unicode-escape'), zip(list(df.byteCode1), list(df.byteCode2)))) 
    return emojis


# In[4]:

def subset_present(df, model):
    # Select only emojis that are in our model (ie in the corpus)
    return df[df["emojis"].map(lambda x: x in model.vocab.keys())]


# In[5]:

emojis_df = loader("./data/emoji_webscraped_expanded.json")
emojis_df["emojis"] = convertEmojis(emojis_df) 
emojis_df_sub = subset_present(emojis_df, emoji_model) # Subset to emojis present in our data (at least 100 times)


# ## Pick one emoji from this list

# In[6]:

for emoji in emojis_df_sub.emojis:
    print(emoji, end="")


# In[14]:

emoji_model.most_similar(positive = ["💔", "💲"], negative= [], topn=10)


# ## Drag and drop emojis here

# In[22]:

emoji_model.most_similar(positive = ['👑', "girl"], negative= [], topn=5)


# ## Examples

# In[15]:

emoji_model.most_similar(positive = ['👑', "girl"], negative= ["guy"], topn=1)


# In[27]:

emoji_model.most_similar(positive = ['🔫'], negative= [], topn=5)


# In[17]:

emoji_model.most_similar(positive = , negative= [], topn=1)


# In[18]:

emoji_model.most_similar(positive = ['🍻'], negative= [], topn=1)


# In[19]:

emoji_model.most_similar(positive = ['🍴'], negative= ["🍺"], topn=10)


# In[20]:

emoji_model.most_similar(positive = ['sport', "🏆"], negative= [], topn=10)


# In[17]:

emoji_model.most_similar(positive = ["snow"], negative= [], topn=10)


# In[18]:

emoji_model.most_similar(positive = ["💲", "💍"], negative= [], topn=10)


# In[ ]:



