## Insight Project

This is a three-week project I did as a Data Science Fellow at Insight Toronto.

### Introduction
Market segmentation is the process of dividing the market into few groups based on features. One segement could be more important for a business than the others. While many large business correctly identify their market segments and target them, it is not obvious for new a new entrepreneur who the customers are. Many new businesses fail due to lack of correct market segmentation.  
### Aim
Identify the population segments and locations in toronto for a business

### Data
* Toronto census data for 140 neighbourhoods
* Data collected by Google places Api for a search query, aggregated at neighbourhood level (140)

### Methods
* Feature selection
* Spearman correlation
* Principal Component Analysis
* Kmeans Clustering

### Web App
http://neighbourhood-finder.ca

App accepts a search query ('chinese restaurant') & Outputs are
* Favourable clusters for the business
* Toronto neighbourhoods colored based on cluster membership & number of businesses in each neighbourhood
* Distribution of important features in different clusters

![Pipeline]
(https://github.com/vineethtm/insight-project/tree/master/images/Slide07.jpg)
