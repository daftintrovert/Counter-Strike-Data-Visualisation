#!/usr/bin/env python
# coding: utf-8

# Hello! I,Vishal Sharma, am the creator of this data-set and just wanted to provide a sample analysis for anyone interested in looking at CS. I look at mainly the pistol round here but many of the techniques can be applied to all types of rounds. There are many ways to analyze this dataset so I hope you can go off and answer interesting questions for yourself :)
# 
# The following questions will be answered in this notebook:
# 
# What are the most common pistol round buys?
# What is the ADR by each pistol on pistol rounds?
# What sites do bomb get planted the most on pistol rounds?
# After bomb gets planted at A/B Site, for all XvX situation, what is the win Probability for Ts?
# In a 1v1, 1v2, 2v1, 2v2, should players play out of site/in-site or one-in one-out to deal the most damage while receiving the least?

# First we will import some basic and essential libraries needed for the analysis

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.misc import imread
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[3]:


df = pd.read_csv(r'C:\Users\hp\Downloads\Compressed\csgo-matchmaking-damage\mm_master_demos.csv',index_col =0)
map_bounds = pd.read_csv(r'C:\Users\hp\Downloads\Compressed\csgo-matchmaking-damage/map_data.csv', index_col=0)


# In[4]:


df.head()


# In[4]:


map_bounds.head()


# Data Prep
# Let's first only isolate for active duty maps as they are the maps that most competitive players really care about. I also want to first convert the in-game coordinates to overhead map coordinates

# In[5]:


active_duty_maps = ['de_cache', 'de_cbble', 'de_dust2', 'de_inferno', 'de_mirage', 'de_overpass', 'de_train']
df = df[df['map'].isin(active_duty_maps)]
df = df.reset_index(drop=True)
md = map_bounds.loc[df['map']]
md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = (df.set_index('map')[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']])
md['att_pos_x'] = (md['ResX']*(md['att_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['att_pos_y'] = (md['ResY']*(md['att_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
md['vic_pos_x'] = (md['ResX']*(md['vic_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['vic_pos_y'] = (md['ResY']*(md['vic_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
df[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']].values


# In[6]:


print("Total Number of Rounds: %i" % df.groupby(['file', 'round'])['tick'].first().count())


# Pistol Round Buys
# Let's first start by taking only pistol rounds and count the number of rounds

# In[7]:


avail_pistols = ['USP', 'Glock', 'P2000', 'P250', 'Tec9', 'FiveSeven', 'Deagle', 'DualBarettas', 'CZ']

df_pistol = df[(df['round'].isin([1,16])) & (df['wp'].isin(avail_pistols))]
print("Total Number of Pistol Rounds: %i" % df_pistol.groupby(['file', 'round'])['tick'].first().count())


# Let's first start by looking at pistol round buys. We infer this from the damage dealt by pistols each round. There is a bias here where if you did 0 damage with that pistol you had, then it doesn't get counted. The potential bias is that aim punch will make most weapons get undercounted but I don't think it's a large issue.

# In[8]:


pistol_buys = df_pistol.groupby(['file', 'round', 'att_side', 'wp'])['hp_dmg'].first()
(pistol_buys.groupby(['wp']).count()/pistol_buys.groupby(['wp']).count().sum())*100.


# Looks like Glock/USP trumps over most pistols.

# Heatmaps of Frequency of Pistol Damage
# Next we can look at what are the most frequent spots when attacking as a T.
# To keep it short, I will just do it on dust2 but changing smap will work on any map within active_duty_maps

# In[9]:


smap = 'de_dust2'

bg = imread(r'C:\Users\hp\Downloads\Compressed\csgo-matchmaking-damage/'+smap+'.png')
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,16))
ax1.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax2.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax1.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
ax2.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
plt.xlim(0,1024)
plt.ylim(0,1024)

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'Terrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='YlOrBr', bw=15, ax=ax1)
ax1.set_title('Terrorists Attacking')

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'CounterTerrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='Blues', bw=15, ax=ax2)
ax2.set_title('Counter-Terrorists Attacking')


# In[10]:


smap = 'de_inferno'

bg = imread(r'C:\Users\hp\Downloads\Compressed\csgo-matchmaking-damage/'+smap+'.png')
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,16))
ax1.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax2.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax1.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
ax2.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
plt.xlim(0,1024)
plt.ylim(0,1024)

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'Terrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='YlOrBr', bw=15, ax=ax1)
ax1.set_title('Terrorists Attacking')

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'CounterTerrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='Blues', bw=15, ax=ax2)
ax2.set_title('Counter-Terrorists Attacking')


# ADR by Pistols
# Next let's take a look at the average damage per round dealt by a player given their pistol. Note that if they had picked up a pistol during the round, it does get counted separately. However, given that most pistol kills are headshots, it shouldn't skew the statistic that much (especially for USPS).

# In[11]:


df_pistol.groupby(['file', 'round', 'wp', 'att_id'])['hp_dmg'].sum().groupby('wp').agg(['count', 'mean']).sort_values(by='mean')


# Deagle has a massive advantage in damage

# Bomb Site Plants
# Let's now look at the Number of bomb plants by site. This statistic tells us the T's preferences for deciding which site to take during the round. Although the possibility of rotates are always there, it gives us a good idea of what to expect.

# In[12]:


df_pistol[~df_pistol['bomb_site'].isnull()].groupby(['file', 'map', 'round', 'bomb_site'])['tick']         .first().groupby(['map', 'bomb_site']).count().unstack('bomb_site')


# Post-plant Win Probabilities by Advantages
# This one could be further disseminated but we want to be able to look at the win probabilities post plant given the context of how many Ts and CTs are alive at that time. First, we can look at overall statistic:

# In[13]:


bomb_prob_overall = df_pistol[~df_pistol['bomb_site'].isnull()].groupby(['file', 'round', 'map', 'bomb_site', 'winner_side'])['tick'].first().groupby(['map', 'bomb_site', 'winner_side']).count()
bomb_prob_overall_pct = bomb_prob_overall.groupby(level=[0,1]).apply(lambda x: 100 * x / float(x.sum()))
bomb_prob_overall_pct.unstack('map')


# ### Well till here we were dealing with the written and mathematical datas behind the various mechanism
# ### taking place inside CS:GO matches.
# 
# ### Now we will plot some graphs and visualize the models and will see what really happens behind a counter strike match.

# In[14]:


#our datas

df.head()


# In[15]:


df.info()


# In[ ]:


#we dont require much af the data and we write them off from the visualization.Datas such as file,date,tick,seconds doesnt
#matter much,at least at this point of time so we drop such column.

#this is called cleaning of data.


# In[16]:


df.head()


# In[17]:


df.columns


# In[18]:


df.drop(['file','date','tick','seconds','hp_dmg','arm_dmg','hitbox','att_id','att_rank','award','vic_id','vic_rank','att_pos_x','att_pos_y','vic_pos_x','vic_pos_y'],axis = 1,inplace = True)
df.head()


# In[19]:


df.drop(['wp_type'],axis = 1,inplace = True)


# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


plt.figure(figsize = (7,7))
sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# Here we can see we dont have much of the bombsite information so we can drop this column since it wont do any good in our visualization

# In[23]:


df.drop(['bomb_site'],axis = 1,inplace = True)


# In[24]:


df.head() #this is our actual real data after being cleaned that we are going to work with


# In[25]:


df.shape


# In[26]:


df.describe()


# In[ ]:


#deleating maps which are not in the active duty pool so that we get better pictures.


# In[27]:


active_duty_maps = ['de_cache', 'de_cbble', 'de_dust2', 'de_inferno', 'de_mirage', 'de_overpass', 'de_train']
df = df[df['map'].isin(active_duty_maps)]
df = df.reset_index(drop=True)


# In[28]:


plt.figure(figsize = (10,12))
sns.countplot(x = 'winner_side',hue= 'map',data = df,palette = 'magma')


# ### Conclusion:-
# 
# ### Dust 2 is T-sided
# ### Cache is T-sided
# ### Inferno is T-sided
# ### and same with rest other maps.
# 
# ### But we can see from the table that winning a round is equally distributed on the maps Overpass and Train.
# ### Interesting...!

# In[29]:


plt.figure(figsize = (10,12))
sns.countplot(x = 'winner_side',hue= 'is_bomb_planted',data = df,palette = 'RdBu_r')


# After planting the bomb there is more probability of CT side to win the round.

# In[30]:


plt.figure(figsize = (10,12))
sns.countplot(x = 'winner_side',hue= 'round_type',data = df,palette = 'coolwarm')


# In[31]:


plt.figure(figsize = (10,12))
sns.countplot(x = 'map',hue= 'round_type',data = df)


# Force buy and eco work well in Mirage..and works worst in inferno or overpass depending upon the frequency of matches you play.

# In[37]:


plt.figure(figsize = (10,12))
sns.countplot(x = 'winner_side',hue= 'round',data = df)


# In[45]:


plt.figure(figsize = (8,10))
sns.boxplot(x = 'map',y = 'avg_match_rank',data = df,palette = 'magma')


# In[47]:


plt.figure(figsize = (10,12))
sns.violinplot(x = 'map',y = 'avg_match_rank',data = df,palette = 'coolwarm')


# In[6]:


sns.lmplot(x = 'round',y = 'avg_match_rank',data = df,hue = 'map')


# In[ ]:




