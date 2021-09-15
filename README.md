Please read before diving into the file. Enjoy!

The code and this file was written by Klint Mane on August 2021 as part of my internship at Overdrive Interactive. 
Note that you would not be able to run the code as the code runs locally. If interested to see how it works and 
deploys please reach out and I would be happy to provide a demonstration. 

The code is divided in three main parts. 

Firstly, we use Ahrefs data to categorize keyword searches automatically. This is done using supervised and
unsupervised machine learning models. Additionally, I also tried using a categorization method that takes
into account the google search ranking and url for that keyword. The best performing model was the 
supervised machine learning model using random forest to do the categorization. Even without optimizing 
the training data by the SEO experts at the company, the model has a 95% accuracy. 

Secondly, we use Pytrends historical data to forecast search volume 1 year into the future. I tried to do 
this using Pycaret and Facebook Prophet. The conclusion was that Facebook Prophet forecasting tool 
outperformed Pycaret on visual inspection in almost all keywords.

Lastly, we needed to put everything together and make it as easy as possible to use the tool internally at
the company. In order to achieve this, I develop a simple app using a platform called Streamlit. 
This makes it possible for the users of the algorithm to simply enter the initial data to be categorized 
from Ahrefs and the historical data using Pytrends. Then they simply enter a username and hit enter.
The code runs and creates an output file with the categorized end results as well as the forecast results. 

The purpose of the app is to save lots of time spent manually categorizing keyword searches. Secondly, 
The algorithm performs much better at a larger scale, leading in better understanding of search volume trends.
Thirdly, it offers a forecast based on the top keywords in each category created above. The following code 
is optimized for the Fortune 50 client, however it can be used on other data as well. All that would need to 
be done is to improve the training data based on the other client keywords and categories.

Click the link to take a look at the outline of the App on Streamlit: ---> https://share.streamlit.io/klintmane1/internship-overdrive-interactive/main/app.py

If any questions or problems arise please reach out at klintmane1@gmail.com. 
