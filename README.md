# ixis-challenge

Imagine that you have been tasked with helping a marketing executive at a large bank understand which characteristics of potential customers are the best predictors of purchasing of one of the bank’s products. Ultimately, the client is interested in using a predictive model to score each potential customer’s propensity to purchase, as well as understanding which customer characteristics are most important in driving purchasing behavior, in order to inform future marketing segmentation personalization, etc.


## Running the code

This project uses [pipenv](https://pypi.org/project/pipenv/) to handle dependencies. Installing this tool is easy, simply run `pip install --user pipenv` on the terminal. 

Once you've cloned this repository and navigated to the project directory, you can install all required packages by running the following
```
$ pipenv install
```

From there, you can run `pipenv shell` to start a new shell subprocess with the relevant dependencies installed (`exit` to deactivate). It's particularly useful to open the Python interpreter inside this subprocess then import functions and packages as usual.

Alternatively, use `pipenv run python [path to python script]` to run any script in the repository.

## Data exploration and wrangling

The first thing I do with any dataset is to simply open it up and see what's inside! Any obvious patterns? How clean does the data look? Any derived features to add? Any junk features to remove? Any missing or corrupted data? How is the data distributed? The best way to get a sense of a dataset this size is to create some visualizations that summarize the data. In this case, I created a stacked bar chart for each attribute showing the frequency of records with each particular value, breaking each bar into two parts to show the proportion of successful subscriptions. 

To generate these charts, navigate to the project directory and run the following command.
```
$ pipenv run python visualize_data.py
```

This script will output a series of charts into a `figures` folder. 

After looking through these graphs, I noticed a few key features which guided the next steps of my work. These findings are explained below. 


### Client contact volume and win rate varies wildly by month

The most unexpected thing I noticed in the dataset was that the total wins in a given month was relatively uniform while the number of client contacts made big swings. This meant that lower-volume calendar months had win rates around 50% while the highest volume months had rates of less than 10%. This is readily seen in the chart below.

<img src="https://user-images.githubusercontent.com/16196888/178872283-abd0c0a5-3b21-4bcc-9fa9-9dc00c8f2465.png" width="500"/>

The documentation accompanying the dataset doesn't offer any clues as to why this might be the case but we could speculate on some reasons this might happen. Do they alternate between targeted campaigns reaching out to rich leads in some months and wider campaings with larger numbers of lower quality leads other months? Are win rates higher in lower volume months because callers had more quality time to focus on each client? Do the salespeople have a win quota each month and they stop making calls once it's hit? 

With more context, I'd try to add a specific feature to measure the root cause. However, we can still leverage our out-of-context observation for predictive power. Since months with a lower volume of clients contacted correspond to a higher probability that those clients are won, it makes sense to associate a feature to each record that measures the total number of contacts in the same month. I did just that by adding a column to the dataset titled `total_clients_contacted_in_month`. Now this *might* be considered cheating since the total number of contacts for that month isn't exactly knowable at the instant a client is contacted but I would think that a business would have a very good estimate for this number that could be submitted in its place. Moreover, in a real scenario we would be able to further investigate this phenomenon and identify the root cause to develop a more precise feature. 

### Previous client history matters 

Another thing that becomes clear when looking at the distribution of each attribute is that a clients prior engagement with the bank matters a lot. The graph below shows that if a client was previously won on a different campaign, they are very likely to convert again on the current campaign. No big surprise there.

<img src="https://user-images.githubusercontent.com/16196888/178883336-ee554ecb-189e-4d24-9c91-0ebb12b57355.png" width="500"/>

Similarly, the variable shown below describes the number of contacts performed before this campaign for the same client. We see that as the number of prior contacts increases, so does the win rate.

<img src="https://user-images.githubusercontent.com/16196888/178883348-568e4a1a-0510-4b0d-86c4-ac5074d4da21.png" width="500"/>

Based on these visualizations, I'd expect these features to be relatively predictive.

### Problematic features

The accompanying documentation mentions that, in regard to the `duration` feature, 

> this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

For that reason, I removed this attribute during cleaning. 

While surverying the data I also noticed a problem with the `pdays` feature. More than 96% of clients had never previously been contacted which meant that they were assigned a junk `pdays` value of `999`. This junk label also contradicted some of the other feaures that came with the dataset. Specifically, there are thousands of records where `pdays = 999` but the `previous` and `poutcome` features both indicate that this client *was* previously contacted. (The `previous` and `poutcome` columns were both consistent with each other.) Given the uncertainty around `pdays`, I decided to drop this attribute when cleaning the data. This seemed particuarly safe to do given that prior contact data was captured better in other features, as described above. 

The charts also showed that a couple attributes had very skewed distributions between their categories, for example, `default` shown below. Only 3 out of the 41,187 contacts were to a client who had credit in default. (The rest are split between "no" and "unknown".) 

<img src="https://user-images.githubusercontent.com/16196888/178883903-4585f6e6-342f-4733-86f8-636a63998f00.png" width="500"/>

I decided to keep this feature (and others like it) around because the other categories did appear to retain predictive information.

### Dealing with categorical attributes

Many of the features included in this dataset are categorical in nature. We need to refactor these features into numerical form to be processed by a machine learning model. A standard way to do that is to use a one-hot encoding, which is what I chose in this case. I used scikit-learn's [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to do this transformation. 

One quirk is that one-hot encoding leads to two redundant columns for binary categorical variables. There are two in this dataset: `contact`, which says whether the call was to a cellular or telephone line, and our target variable `y` which is a yes/no field indicating won/lost. I treated these separately and did the natural thing of resolving them to a single column. 


## Model training and testing

## Conclusion and next steps

## Client deliverable
