Haplophysh
==========

### What is it?
Experimenting with some convolutional and recurrent neural net architectures with word and character embeddings to 
detect phishing URLs. This is the project I did for Network Security course in winter 2020. You can check out a one-page extended abstract which summarises the key points of this project [here](ExtendedAbstract.pdf).

### What did you learn?
First thing I learned is that *it works*. Phishing pages DO give themselves away in their URLs alone. 

Also I learned how important it is that the model size (number of trainable parameters / its learning capacity) and the data set size to be proportionate. One of the models that combines character level and word level embeddings is almost the same 
architecture that URLNet proposed but it is way smaller because my data volume was orders of magnitude smaller. It still manages to perform pretty well sometimes surpassing URLNet.

These two were the highlights for me but there are many things to learn doing such a project, listed below. I had already learned many of them while doing my Masters' thesis; yet it is worth mentioning that these can be other take aways of this project. If anyone wants to see some examples of how to do these things in Keras and TensorFlow 2 they can take a look at this repository!

- How embeddings word
- How to deal with 1-D convolutions for sequence processing
- How to use LSTM/GRU recurrent layers for sequence processing
- How to tune the hyper parameters such as the layer sizes or batch size
- How to get the most out of the GPU in training
- How to use dataset generators to deal with larger datasets more easily
- How to deal with model architectures with multiple inputs


### Where to get the data from?
If you want to do a big research on it you'd be better off to collaborate with a large corporation or security group. 
But if it is not possible for your situation I'd recommend these data sources which I used myself, in no particular order:

- [Malware URL](https://www.malwareurl.com/) you need to negotiate a price, they were kind enough to provide it to me for this course project for free. Kudos to them!
- [Phish Tank](https://www.phishtank.com) updates every 6 hours
- [This data set from 2016](https://www.unb.ca/cic/datasets/url-2016.html) from Canadian Institute for Cybersecurity. [Mamun '16]
- [This Kaggle data set](https://www.kaggle.com/teseract/urldataset)
- [This UCI data set](http://archive.ics.uci.edu/ml/datasets/URL+Reputation)


I cannot share them on github. First because they get out of date quickly (in a matter of hours!). Second, there won't be any point in mirroring them in a repository too, most of them are publically available. Also, I might not have the permission to share some of them (I'm sure about the first one; redistribution is a no no). And at last, for now I have no plans for keeping this repository up to date in long term; I have completed a project and had my take-aways, I'm done with it for now. 


### Related Work
- Benavides, E., Fuertes, W., Sanchez, S., & Sanchez, M. (2020). [Classification of Phishing Attack Solutions by Employing Deep Learning Techniques: A Systematic Literature Review.](https://doi.org/10.1007/978-981-13-9155-2_5) In Developments and Advances in Defense and Security (pp. 51-64). Springer, Singapore.
- Le, H., Pham, Q., Sahoo, D., & Hoi, S. C. (2018). [URLnet: Learning a URL representation with deep learning for malicious URL detection.](https://arxiv.org/abs/1802.03162) arXiv preprint arXiv:1802.03162.
- Selvaganapathy, S., Nivaashini, M., & Natarajan, H. (2018). [Deep belief network based detection and categorization of malicious URLs.](https://doi.org/10.1080/19393555.2018.1456577) Information Security Journal: A Global Perspective, 27(3), 145-161.   
- Saxe, J., & Berlin, K. (2017). [eXpose: A character-level convolutional neural network with embeddings for detecting malicious URLs, file paths and registry keys.](https://arxiv.org/abs/1702.08568) arXiv preprint arXiv:1702.08568.
- Eshete, B., Villafiorita, A., & Weldemariam, K. (2012, September). [Binspect: Holistic analysis and detection of malicious web pages.](https://doi.org/10.1007/978-3-642-36883-7_10) In International Conference on Security and Privacy in Communication Systems (pp. 149-166). Springer, Berlin, Heidelberg.
- Ma, J., Saul, L. K., Savage, S., & Voelker, G. M. (2009, June). [Beyond blacklists: learning to detect malicious web sites from suspicious URLs.](https://doi.org/10.1145/1557019.1557153) In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1245-1254).

### Misc.
*What does this name mean anyways?*
It's a dull play on words. Haplophryne is an \[ugly\] fish living in deep ocean. *physh*, *phish*, *deep*, *deep*learning, get it? Okay I'll stop. Sorry.  

*Why is everything on master branch? Do YoU EvEn GIT BrUh?*
Sorry, I do. I was just lazy here, don't judge.  

*Are you publishing it?*
I don't know. I wrote a 10 page conference paper as a deliverable for the course, but it needs more work before being ready for publication. A summary of that paper, as an extended abstract, is now available [here](ExtendedAbstract.pdf).

*I read the abstract, who is "We" exactly?*
I did the project by myself. I used "we" throught the paper and the EA out of habit.