
# Machine Learning for video transcoding verification in Livepeer’s ecosystem (I)

Livepeer is a protocol for video transcoding. Transcoding is a compute-heavy process that has traditionally carried a high technical and monetary cost. Livepeer aims to slash the cost of transcoding with an open network that incentivizes transparent competition among suppliers of transcoding capacity. The rules of the Livepeer protocol are backed by smart contracts on the Ethereum blockchain, an immutable ledger. The combination of an open, permissionless network and immutable non-reversibility attracts byzantine (adversarial) behavior: any participant can attempt to “break the protocol” without immediate consequences from within the Livepeer network. A verification mechanism is necessary to decide whether the transcoding work was done correctly.

In a quest to define an algorithm that would help [Livepeer](https://livepeer.org/) include such mechanism, we at [Epiclabs](https://www.epiclabs.io/) are making some progress (see [article I](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea), [article II](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380) and [article III](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-iii-744ba1c1d5d7)).

Summarizing, in the [first article](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-f66f724b2aea) were introduced the **difficulties associated with common video metrics (PSNR, VMAF, SSIM and MS-SSIM). **There we saw the limitations of simply aggregating time series of these metrics into an average value. In [article II](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-ii-6827d093a380) we presented alternative ways to tackle the associated oversimplifications. We introduced **richer time-series aggregators such as the euclidean distance and also some alternative metrics** that looked at differences between consecutive frames. Finally, in [article III](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-iii-744ba1c1d5d7) we **defined a rich set of five metrics that would look more carefully at specific characteristics of video frames**.

In this first article of the series about machine learning we will explain how all these metrics can be used as inputs to a Binary Classifier and how they are being integrated within Livepeer’s verification process.

## Supervised or unsupervised learning?

The answer: both.

In the problem at hand, our classifier is meant to establish whether a video encoding is done correctly. In the issues section of Livepeer’s repo ([here](https://github.com/livepeer/verification-classifier/issues/1)) we read a more accurate description of what does that mean:
> *Some possible attacks or unreasonable flaws in the video may include:*
> - Bitrate (eg, 720p output should not have a higher bitrate than an 1080p source […])
> - Rotations
> - Watermarking, vignetting
> - Color shifting, excessive color correction, filters
> - Low quality encodes

Extensive as it might seem, the list has to remain open, as the creativity of human beings is endless, more so when being part of a community and the incentive is high. Our first approach was to naïvely create a large dataset of video encodings containing all the attacks in the list above, plus some others that we could imagine (full list is available [here](https://github.com/livepeer/verification-classifier/blob/master/scripts/README.md)). Soon it became apparent we were missing something: how many more attacks are we not considering?

![Rumsfeld’s completed table of epistemic uncertainty: “There are known knowns. There are things we know that we know. There are known unknowns. That is to say there are things that we now know we don’t know. But there are also unknown unknowns. There are things we do not know we don’t know.” — D. Rumsfeld](https://cdn-images-1.medium.com/max/2000/1*p8bQiTq3E0HlAu0eRv-FlA.png)

*Rumsfeld’s completed table of epistemic uncertainty: “There are known knowns. There are things we know that we know. There are known unknowns. That is to say there are things that we now know we don’t know. But there are also unknown unknowns. There are things we do not know we don’t know.” — D. Rumsfeld*

By definition, when training a model through supervised learning for classification we must supply, for each training sample, with the category to which the sample belongs to (its label). This forces us to define beforehand all kinds of possible attacks.

Enter unsupervised learning.

By contrast, training a classification model in an unsupervised manner basically means that the labels of the categories are not available. Unsupervised learning techniques are very useful when there is a large number of samples for a particular class but very few of the others (one class problems). This is the case, for example, when an engine is operating for hours and hours undisturbed. Until it breaks down. If we wanted to classify its performance by simply observing its every minute of running, we would have a lot of samples labeled as “it’s working”, but very few “it’s broken” samples. Similarly, the labels available when bank or insurance fraud is being perpetrated are hard to obtain, as frauds basically work by exploiting vulnerabilities in the systems they attack that the designers weren’t able to foresee.

In the absence of such valuable source of information, all we can resource to is *normality*. By statistical inspection of the supplied training data, unsupervised learning techniques are able to extract common patterns of what is *normal* behavior. Everything that is not *normal* can be regarded as *novel*, as an *outlier*. Eventually, the outliers are determined from the “closeness” of vectors using some suitable distance metric.

…and finally, semi-supervised.

Half way between supervised and unsupervised learning lies the semi-supervised learning paradigm. In this case, as in unsupervised learning, what we will be modeling is only the normal behaviors. But alas, we also have access to the labels. As we decided to keep the categorization simple (attack / no attack), we just need to remove those labels and let the model figure out what is the boundary around the set of features we have created. We will use them for testing, though, but not in the training stage.

In order to obtain the images below we have employed two popular unsupervised learning techniques, namely T-SNE (T-Distributed Stochastic Neighbour Embedding) and Random Projections. We have applied them over a data set obtained from the features described in previous stories. Blind as they are to the labels, we can still cheat and color them for visualization purposes (the hand drawn boundary basically means that the cluster are separable, which is great news!). In a latter article, we will enter into implementation details (or you can have a glance in [our repository](https://github.com/livepeer/verification-classifier/tree/master/machine_learning)), but for now, let’s just mention that the output of a Random Projections model is used to feed our classifier. The advantage of this step is that we can narrow down the number of inputs (called dimensionality reduction) not only without loss of information, but more like enhancing certain aspects of our data set.

![](https://cdn-images-1.medium.com/max/2732/1*sMQ7ICGkC8mXXqKoLSZ0Tg.png)

![Dimensionality reduction of 20.000 specimens of our data set by means of T-SNE (above) and Random Projections (below). By changing our “perspective” with respect to our data we can better comprehend it. Fortunately for us, at the time of training our model we can count on the labels to distinguish bad renditions made by ourselves. However, in production such information is not available (hence the need of our classifier in the first place).](https://cdn-images-1.medium.com/max/3666/1*rNBamqaSl1OWZ5O7xTfDFg.png)*Dimensionality reduction of 20.000 specimens of our data set by means of T-SNE (above) and Random Projections (below). By changing our “perspective” with respect to our data we can better comprehend it. Fortunately for us, at the time of training our model we can count on the labels to distinguish bad renditions made by ourselves. However, in production such information is not available (hence the need of our classifier in the first place).*

Moreover, we will prop up our system with yet another supervised learning classifier, so that new kinds of attacks can be learnt from and gain hindsight from the system’s own experience. This will shape the basis for our meta-model, combining the best of all worlds.

Cool huh? At this point, hopefully, the advantage of our approach becomes evident: we can create a system that is robust to “any” kind of attack, even those “that we don’t know that we don’t know”.

## Describing a video asset. What makes a good encoding?

This is great then. We have a strategy. But now we need to supply our learning model with enough information. The kind of information that helps it to figure out what makes it a “good” rendition so everything else can be categorized as garbage. In our previous story ([article III](https://medium.com/@epiclabs.io/assessing-metrics-for-video-quality-verification-in-livepeers-ecosystem-iii-744ba1c1d5d7)) we described a set of computable features that seemed to describe fairly well the intrinsic characteristics of each one of our video samples. They describe inter-frame differences between an original and its renditions for:

* **Color** — Inter-Frame Histogram Distance (IFHD)

* **Contours** — Inter-Frame Contour Difference (IFCD)

* **Energy** — Inter-Frame Discrete Cosine Transform Difference (IFDCTD)

* **Textures** — Inter-Frame normalized cross-correlation (IFNCC)

* **Volume** — Inter-Frame Low Pass Difference (IFLPFD)

We also have a large enough data set created from the YT8M’s data set with original videos and different renditions. Instructions and utilities to access those 10 second video segments are available in the [project’s repo](https://github.com/livepeer/verification-classifier). With an 80% train / 20% test split, we have **102531** train specimens and **25633** test specimens. Our positive sample pool (i.e. “good” renditions), is made of those videos transcoded at the bitrate prescribed by Youtube according to their resolution (144p, 240p, 360p, 480p, 720p). The rest are attacks and belong to the negative pool.

This leaves us with **14550** positive training specimens and **87981** negative training specimen, from where we will only take **14550** in order to maintain a balanced dataset. Hopefully they will provide enough diversity to make a good generalization. The csv file containing the full dataset can be found [here](https://storage.googleapis.com/feature_dataset/data-large.tar.xz).

For the generation of this dataset, we have computed all five features enumerated above and obtained a time series for each rendition. We have obtained their **maximum**, **mean** and **standard deviation** values. Then we have calculated the **euclidean distance** and the **Manhattan distance** between these time series and that of the original. Finally, for each and every sample we have also added:

* **resolution** (vertical size of the video, in pixels)

* **fps** (number of frames per second of the video)

* **size** (in memory, in bytes)

This set of features is hopefully rich enough to describe in numerical terms all that a model needs to infer whether a transcoded video is a good or a bad copy of another. A sandbox notebook for exploring the properties of the metrics is provided [here](https://github.com/livepeer/verification-classifier/blob/master/feature_engineering/notebooks/Spatial_temporal_activity.ipynb).

## Learning machines

At last, with no more preambles, we can introduce the different techniques we have chosen to detect those nasty guys.

From the **Semi-supervised Learning** side, we have invited to our party to One Class Support Vector Machine ([**OCSVM**](https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html)**) and Isolation Forest ([**IF**](https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html)**).

From the **Supervised Learning** category, we have decided to count on Random Forest ([**RF**](https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py)**), AdaBoost ([**AB**](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html)**), Support Vector Machine ([**SVM**](https://scikit-learn.org/stable/modules/svm.html)**, not to be mistaken with **OCSVM**), and XGBoost ([**XGB**](https://xgboost.readthedocs.io/en/latest/)**).

Finally, let’s not forget about our hybrid **meta-model**, combining with a logical AND the best resulting supervised learning model with the best unsupervised one. The supervised learning model is trained in recognizing those attacks where the unsupervised performs the worst. This makes for a very robust system where we remain agnostic to unknown unknowns while exploiting the advantages of what we already know.

We will not enter here in details about their implementations. They are available in the links provided and also in [our repository](https://github.com/livepeer/verification-classifier/tree/master/machine_learning/notebooks). We will, instead, display our results with the aforementioned data set and using those techniques. The choice is fairly arbitrary and a better informed reader might suggest better methods. For us, the trade-off between ease of implementation and accuracy has been the main drive.

### Our definition of success

But first, let’s define how we will be grading our models. The image below is illustrative enough of what all binary classifications are about.

![A binary classification of sheep into light and not light. There is a total population of 20 sheep, of which 9 are truly light and 11 truly dark.](https://cdn-images-1.medium.com/max/2000/1*Ar03nm4nRKaz27Z1ObGMZw.png)

*A binary classification of sheep into light and not light. There is a total population of 20 sheep, of which 9 are truly light and 11 truly dark.*

If the results in the picture were the job of a mathematical hound trained to group sheep into light colored and not light colored, we could say he accomplished fairly well the task (for a dog). There is a total of 20 heads of cattle, of which 11 are dark and 9 can be said of as light colored. One of them however stands among the wrong bunch in the bright side (a false positive) and three of them disguise themselves poorly in the darkness of the negative crowd (three false negatives). In total, our shepherd dog misplaced *only* four out of twenty sheep (an 80% accuracy, not bad).

Now imagine the owner of the sheep wants to sell the wool in the market and has to pay a truck driver for each ruminant, but there is only demand for clear colors this season. Every miss would cost some money that our herdsman might not be willing to give up. Moreover, if instead of a flock as balanced as this one there would be several hundreds of dark sheep and only a few bright ones, the measure of accuracy alone would not be sufficient.

Our rancher needs a system where the ratio between true positives and all specimens (call it sensitivity, recall or TPR) and the ratio between true negatives and all specimens (call it specificity, or TNR) are correctly weighed. In the picture, we have 6 true positives (TP), 1 false positive (FP), 7 true negatives (TN) and 3 false negatives (FN). The F-score is a mathematical tool that weighs the precision with the recall. From [here](http://www.marcelonet.com/snippets/machine-learning/evaluation-metrix/recall-and-precision), we can better understand the meaning of these magnitudes:
> A quick definition of recall and precision, in a non-mathematical way:
> ***Precision:*** high precision means that an algorithm returned substantially more relevant results (total actual positives) than irrelevant ones
> ***Recall:*** high recall means that an algorithm returned most of the relevant results (total actual positives)

In our case of study, relevance is determined by the fact that all the fleece that is dark might not be sold this year, so we want to maximize the amount of light wool. Precision is computed as:
> precision = TP / (TP + FP) = 6 sheep / (6 sheep + 1 sheep) = 85,7%

while the recall is:
> recall = TP / (TP + FN) = 6 sheep / (6 sheep + 3 sheep) = 66,6%

Finally, the F1-score of our algorithmic shepherd dog is:
> F1 = 2 * (precision * recall) / (precision + recall) =74,9%

So, F1-score has resulted a bit lower than accuracy (measured above as 80%). But what does this tell us? Well, now we can train our shepherd dog keeping in mind that recall needs to be pushed up a bit, because false negatives have an economic impact we would like to avoid. Moreover, imagine the flock had 100 dark sheep and only two light colored. By simply putting every sheep in the dark group our dog would achieve an accuracy of 99,0%, but the recall would be a total failure.

In Liveper’s particular application, we call true positives (**TP**) to all those videos that, being **correctly** transcoded, have been **correctly** identified as such by our model. Similarly, true negatives (**TN**) are those renditions that being **wrongly** transcoded have been **correctly** classified as such. On the other hand, we have false positives (**FP**) wherever a **wrongly** transcoded rendition has been labelled as **correctly** transcoded by our model. And finally, we have false negatives (**FN**) wherever a **correctly** transcoded rendition is tagged as **wrongly** transcoded.

Given the nature of Livepeer’s business, we have to be extremely careful with the first kind (*TP*). The system doesn’t want to penalize those who, acting in good faith, may be mistakenly considered as bad. The main reason is that the proof of stake system would slash them and money would be destroyed.

As a matter of fact, there is even certain tolerance to what can be *smuggled* as good while being bad (FP) because the system has other *organic* mechanisms than can deal with it. When a broadcaster identifies a misbehaving transcoder, chances are high that there will be no more business done and reputation will be lost (but not money). In other words, we want a False Negative Rate (*FNR*) as close to zero as possible. We need a high recall.

## Results

In order to guide our steps in the process of training our models, we have chosen a F-Beta score metric. F-Beta score is defined as:

![](https://cdn-images-1.medium.com/max/2000/1*fSeFmv7Guhs95pgq86evdA.png)

When the Beta factor is equal to 1, we have the F1-score, which is basically the mean between precision and recall as we saw in our sheep illustrative example. However, for the problem at hand we wanted to give more weight to the recall, because we don’t want to punish unfairly those good transcoding jobs.

What follows is the evolution in time (as sprints of our work in Livepeer) of the F20 value for each of the models we have developed:

![F20 score evolution of each of the machine learning techniques implemented in the Livepeer verification classifier. On each sprint we explored different features and characteristics of the metrics, achieving different results. At the time of writing this story, a Supervised Learning Support Vector Machine (SVM) has the highest score. From the Unsupervised Learning category, the Autoencoder hold this prevalent position.](https://cdn-images-1.medium.com/max/2000/1*MFIjw7c6Ob1_pTKfNeskpw.png)

*F20 score evolution of each of the machine learning techniques implemented in the Livepeer verification classifier. On each sprint we explored different features and characteristics of the metrics, achieving different results. At the time of writing this story, a Supervised Learning Support Vector Machine (SVM) has the highest score. From the Unsupervised Learning category, the Autoencoder hold this prevalent position.*

Further explanation about these results belongs to another story in this series. The evolution of each model is complex and needs details to be well understood.

*Grosso modo*, at different sprints we have tested and explored different feature combinations, different video segment lengths and different frame rescaling sizes. The purpose of this search is to find a good trade-off between accuracy and speed of inference.

## Conclusions

We have presented the dataset and mindset that surrounds a complex problem like classifying video transcoding attacks. The main issue in this case of classification problems is the epistemic uncertainty associated, as there is no way to forecast what kind of attacks are possible.

We use different machine learning techniques from both the supervised and unsupervised learning families to solve our problem.

We have explained the need of a specific measure for the accuracy, the F-Beta score, given the specifics of Livepeer’s operation.

Finally, we have presented our results in terms of this score for each of the techniques.

### About the authors

[Rabindranath](https://www.epiclabs.io/member/rabindranath/) has a PhD in Computational Physics by the UPC and AI researcher. [Dionisio](https://www.epiclabs.io/member/dionisio/) is Computer Science Engineer by the UPM specialized in Media. [Ignacio](https://www.linkedin.com/in/ignacio-peletier-ribera/?locale=en_US) is a Telecommunications Engineer specialized in Data Science and Machine Learning. They are part of [Epic Labs](https://www.epiclabs.io/), a software innovation center for Media and Blockchain technologies.

[Livepeer](https://livepeer.org/) is sponsoring this project with Epic Labs to research ways to evaluate the quality of video transcoding happening throughout the Livepeer network. This article series helps summarize our research progress.
