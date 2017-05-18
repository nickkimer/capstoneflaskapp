# Building a Research Assistant Management Platform Utilizing Natural Language Processing
## Nicolas Kim, Matthew DaVolio, Joel Stein, and Rafael Alvarado

### Live View
http://datascience.shanti.virginia.edu/ramp/home

### Project Abstract
As the largest navy in the world, the United States Navy must quickly retrieve and communicate information across the globe in order to protect the country and its allies. As the amount of information that analysts must sort through grows exponentially, this creates a more difficult environment for extracting useful and relevant data. This problem of information complexity in research is present in other disciplines, creating a demand for a system to increase both effectiveness and efficiency of research. In this paper, we propose a system, the Research Assistant Management Platform (RAMP), to assist analysts in their duties in retrieving relevant maritime information. The platform is a realization of a proof of concept utilizing a database backend for storage of Requests for Intelligence, real-time updating topic models for document similarity and thematic analysis, and an intuitive front-end user interface with built-in work-flow operations such as saving and model visualizations. The system provides effective information retrieval and an integration of a previously fragmented multi-step process into one research platform. The prototype uses surrogate data which is contextually and structurally similar to metadata which would be seen by analysts. With this data, the system was evaluated based on user testing and the determined relevance of the displayed results from a query. The final topic model consists of fifty topics which allow analysts to improve their responses and learn from the existing corpus of RFIs. To the best of our knowledge, the implementation of an integrated research platform in this context is a novel application and although presented in the context of maritime research, this platform is generalizable to other commercial and academic uses.

### Data Collection
Data was collected using Python's Praw package to scrape data from the subreddit AskHistorians (reddit.com/r/askhistorians). The scrape was done based on the search keywords "Navy", "Ship", "Port" and "Military". The data was stored in an SQLite table consisting of just over 7,000 observations. The data was then cleaned in a Python pipeline performing standard NLP cleaning techniques. NLTK was utilized for several of these cleasing steps.

### Model Creation
The topic models were created with the use of Python's Gensim package. A MM corpus was created along with a dictionary object and index which was then transfered into a numpy array. A Latent Dirichlet Allocation Model was used to create the topic model. 30 Topics were created in the model (other numbers tested include 40, 50, 100). Also a Word2Vec model was created using Gensim as well.

### Web App
The web app was built with the use of Python's Flask package and relied heavily on javascript. The app allows users to perform Document-to-Document similarity searches with a query, save documents of interest that appear in these results, view a lit of topics with the top 5 words in each topic, view a network diagram of the topics, and view the corpus through an intertopic distance map.
#### To Run Web App Locally
Web app is run with `python run.py`

