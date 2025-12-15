# Information Theory
## Information Entropy
$$\mathtt{H}(X) = -\sum \Pr[X=e] \log \Pr[X=e]$$

Measures how "surprising" or "uncertain" a situation is.



## Relative Entropy (KL Divergence)
$$\mathtt{KL}(Y, X)=\sum \Pr[Y=e]\log \frac{\Pr[Y=e]}{\Pr[X=e]}$$

Measures how badly one probability distribution approximates another. 

Note that the KL divergence is **asymmetric**: $\mathtt{KL}(Y,X)$ is generally not equal to $\mathtt{KL}(X,Y)$.

## Jerrreys Divergence

$$\mathtt{J}(Y,X) = \mathtt{KL}(Y, X) + \mathtt{KL}(X,Y)$$

Symmetric version of KL divergence, like a "friendly average" of both directions.

## Cross Entropy
$$\mathtt{H}(Y,X)  = -\sum \Pr[Y=e]\log \Pr[X=e] = \mathtt{H}(Y) + \mathtt{KL}(Y, X)$$

$$\mathtt{H}(X,Y)  = -\sum \Pr[X=e]\log \Pr[Y=e] = \mathtt{H}(X) + \mathtt{KL}(X, Y)$$


Note that the cross entropy is **asymmetric**: $\mathtt{H}(Y,X)$ is generally not equal to $\mathtt{H}(X,Y)$.

### What is cross entropy intuitively?
Cross entropy measures the average number of bits needed to represent events from one probability distribution using the optimal code for another distribution.


> **Information Entropy:** probability distribution of $X$ → optimal codewords for $X$ → encode data from $X$.
>
> **Cross Entropy $H(Y,X)$:** probability distribution of $X$ → optimal codewords for $X$ → encode data from $Y$.
>
> **Cross Entropy $H(X,Y)$:** probability distribution of $Y$ → codewords for $Y$ → encode data from $X$.


## Conditional Entropy
$$\mathtt{H}(Y|X) = \sum \Pr[X=x] \mathtt{H}(Y|X=x)$$
The remaining uncertainty after you know something.

# Feature & Label
Let $X$ denote features and $Y\in \{Good, Bad\}$ denote labels, and let $b$ and $g$ be the total number of bad and good samples, respectively. For the $i$-th bin, $b_i$ and $g_i$ represents the counts of bad and good samples in that bin.


| $X$|  $Y$ | | | $X_1$ |$Y=Good$ | | | $X_0$ | $Y=Bad$  |
|:--:|:----:|-|-|:-----:|:-------:|-|-|:-----:|:--------:|
| 1  | Good | | |   1   |   Good  | | | ~~1~~ | ~~Good~~ |
| 1  | Bad  | | | ~~1~~ | ~~Bad~~ | | |   1   |    Bad   |
| 4  | Bad  | | | ~~4~~ | ~~Bad~~ | | |   4   |    Bad   |
| 3  | Bad  | | | ~~3~~ | ~~Bad~~ | | |   3   |    Bad   |
| 3  | Good | | |   3   |   Good  | | | ~~3~~ | ~~Good~~ |
| 2  | Bad  | | | ~~2~~ | ~~Bad~~ | | |   2   |    Bad   |

## WoE (Weight of Evidence)

$$\mathtt{WoE}_i=\log \frac{g_i\mathbin{/}g}{b_i\mathbin{/}b} = \log \frac{\Pr[X=x_i|Y=Good]}{\Pr[X=x_i|Y=Bad]}$$

## IV (Information Value)
$$\mathtt{IV}(X,Y) = \sum \mathtt{IV}_i = \sum (g_i\mathbin{/}g-b_i\mathbin{/}b)\mathtt{WoE}_i = \sum (g_i\mathbin{/}g-b_i\mathbin{/}b)\ln\frac{g_i\mathbin{/}g}{b_i\mathbin{/}b}$$


### Deriving IV from Jeffreys divergence.
$$\mathtt{IV}(X,Y)=\mathtt{J}(X_1, X_0)\ln2$$

**IV is essentially Jeffreys divergence** between the distribution of feature $X$ for **bad cases** ($X_0$, where $Y=0$) and its distribution for **good cases** ($X_1$, where $Y=1$). 

>$X_0=(X|Y=0)$ and $X_1=(X|Y=1)$, then we have $\Pr[X_0 = x_i] = \frac{b_i}{b}$ and $\Pr[X_1 = x_i] = \frac{g_i}{g}$.
>
> $\mathtt{KL}(X_1\| X_0) = \sum \Pr[X_1=x_i] \log \frac{\Pr[X_1=x_i]}{\Pr[X_0=x_i]}=\sum g_i/g\log \frac{g_i/g}{b_i/b}$
>
> $\mathtt{KL}(X_0\| X_1) =\sum b_i/b\log \frac{b_i/b}{g_i/g}  = - \sum b_i/b\log \frac{g_i/g}{b_i/b}$
>
> $\mathtt{J}(X_1, X_0) = \mathtt{KL}(X_1\| X_0) + \mathtt{KL}(X_0\| X_1) = \sum (g_i/g-b_i/b)\log \frac{g_i/g}{b_i/b}$


### Why does a high IV signify strong predictive power?
$\mathtt{IV}(X,Y)$, i.e., $\mathtt{J}(X_0,X_1)$, quantifies **how well feature $X$ separates the good and bad cases**. A larger IV indicates a greater ability to discriminate between the two classes, i.e., stronger predictive power.​

# Label & Predict Label
|  $Y$ |Score | | $\hat Y$ (threshold=0) |$\hat Y$ (threshold=0.23)|$\hat Y$ (threshold=0.81)|
|:----:|:----:|-| :--------------------: | :---------------------: | :---------------------: |
| Good | 0.23 | |         Good           |         Bad             |         Bad             |
| Bad  | 0.59 | |         Good           |         Good            |         Bad             |
| Bad  | 0.81 | |         Good           |         Good            |         Bad             |
| Bad  | 0.15 | |         Good           |         Bad             |         Bad             |
| Good | 0.77 | |         Good           |         Good            |         Bad             |
| Bad  | 0.18 | |         Good           |         Bad             |         Bad             |
## Confusion Matrix

### TP/TN/FP/FN
True Positive / True Negative / False Positive / False Negative
### Accuracy/Precision/Recall/Specificity/F1-Score

Accuracy = (TP+TN)/(P+N)

Precision = TP/(TP+FP)

Recall = TP/(TP+FN)

Specificity = TN/(TN+FP)

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## ROC
ROC (Receiver Operating Characteristic Curve) is defined as the set of points of **FPR and TPR pairs** obtained by varying the classification threshold.

> threshold + Score → $\hat Y$
>
> $Y$ + $\hat Y$ → FPR, TPR

## AUC
AUC is the area under the ROC curve.
