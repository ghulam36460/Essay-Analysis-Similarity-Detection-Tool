# %%
import os
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
def getessays(folder="txt files"):
    allfiles = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            try:
                with open(path, 'r', encoding='utf-8') as myf:
                    allfiles[file] = myf.read()
            except:
                with open(path, 'r', encoding='latin1') as myf:
                    allfiles[file] = myf.read()
    return allfiles

# %%
def basicstats(esdata, folder="txt files"):
    info = {}
    for filename, content in esdata.items():
        allwords = re.findall(r'\b\w+\b', content.lower())
        puncs = re.findall(r'[^\w\s]', content)

        info[filename] = {
            "wordcount": len(allwords),
            "charcount": len(content),
            "charnospace": len(content.replace(" ", "")),
            "linecount": len(content.splitlines()),
            "avgwordlen": sum(len(w) for w in allwords) / len(allwords) if allwords else 0,
            "essaylenb": os.path.getsize(os.path.join(folder, filename)),
            "longword": max(allwords, key=len) if allwords else "",
            "mfreqword": Counter(allwords).most_common(1)[0][0] if allwords else "",
            "uniqwordcount": sum(1 for c in Counter(allwords).values() if c == 1),
            "punctcount": len(puncs),
            "uppercount": sum(1 for ch in content if ch.isupper()),
            "lowercount": sum(1 for ch in content if ch.islower())
        }
    return pd.DataFrame(info).T


# %% [markdown]
# ## Cell 4

# %%
def uniqrat(esdata):
    out = {}
    for file in esdata:
        txt = esdata[file]
        wrds = re.findall(r'\b\w+\b', txt.lower())
        if wrds:
            count = Counter(wrds)
            uniq = sum(1 for w in count.values() if w == 1)
            out[file] = uniq / len(wrds)
        else:
            out[file] = 0
    return pd.Series(out, name="uniq_ratio").sort_values(ascending=False)


# %% [markdown]
# ## Cell 5

# %%
def worddiv(esdata):
    div = {}
    for file in esdata:
        txt = esdata[file]
        wrds = re.findall(r'\b\w+\b', txt.lower())
        if wrds:
            div[file] = len(set(wrds)) / len(wrds)
        else:
            div[file] = 0
    return pd.Series(div, name="worddiversity").sort_values()


# %% [markdown]
# ## Cell 6

# %%
def strsimilar(esdata):
    vect = CountVectorizer().fit_transform(esdata.values())
    cos = cosine_similarity(vect)
    fnames = list(esdata.keys())
    return pd.DataFrame(cos, index=fnames, columns=fnames)


# %% [markdown]
# ## Cell 7

# %%
myessays = getessays()
mystats = basicstats(myessays)
myuniqrat = uniqrat(myessays)
myworddiv = worddiv(myessays)
mysim = strsimilar(myessays)

print("== Essay Stats ==")
display(mystats)

print("\n== unique ratio ==")
display(myuniqrat)

print(f"\n essay with most unique words: {myuniqrat.idxmax()}")

print("\n== word diversity (small = more repeat words) ==")
display(myworddiv)

print(f"\n most repeated words in: {myworddiv.idxmin()}")

print("\n== cosine sim matrix ==")
display(mysim)


# %% [markdown]
# ## Cell 8
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sim2 = mysim.copy()
for i in range(len(sim2)):
    sim2.iloc[i, i] = 0

simtotal = sim2.sum(axis=1).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=simtotal.values, 
    y=simtotal.index, 
    hue=simtotal.index,         
    palette="cubehelix", 
    legend=False                 
)
plt.title("\n Similarity score of each essay\n")
plt.xlabel("\nSimilarity Index total\n (eaasy with most similarity score is most similar to others)\n")
plt.ylabel("File name")
plt.tight_layout()
plt.show()


