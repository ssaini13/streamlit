#Necessary imports
import streamlit as st
import numpy as np 
import pandas as pd 

from wordcloud import WordCloud
from tqdm.notebook import tqdm
from collections import Counter
from textblob import TextBlob
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import base64
from PIL import Image
from bertopic import BERTopic

nltk.download('stopwords')

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('cesar-couto-27HiryxnHJk-unsplash.jpg')


#image = Image.open('istockphoto-1420864052-612x612.jpg')
#st.sidebar.image(image, caption='Natural Language Processing')


#Headings for Web Application
st.sidebar.title("NLP Web App")
st.sidebar.image("istockphoto-1420864052-612x612.jpg")
option = st.sidebar.selectbox('Select Service',('Sentiment Analysis', 'Topic Modeling'))


#option is stored in this variable

if option == 'Sentiment Analysis':
    st.subheader("Sentiment Analysis") 
    
    st.write("Sentiment analysis is contextual mining of words which indicates the social sentiment of a brand and also helps the business to determine whether the product which they are manufacturing is going to make a demand in the market or not.")
    
    col1, col2 = st.columns([1, 1])
    with col1: 
        st.image("istockphoto-1291779592-170667a.jpg") 
    with col2: 
        st.write("""Our approach has been to first of all scrape the data from the Glassdoor website and extract the company reviews data for various organisations, post which we have used a “Roberta-base-sentiment” transformer from hugging face to rank the reviews on the basis of their polarity.""")
        #upload file
    df = pd.read_excel('Glassdoor_reviews_data.xlsx', sheet_name = 'main')

    company = df['Company'].unique()


    # In[39]:


    dict1={}
    star_reviews=[]
    for i in company:
        dict1.update(df['Ratings'][df['Company'] ==i].value_counts())
        star_reviews.append(dict1[5])    


    # In[80]:


    fig = px.bar(df, x=company, y=star_reviews, labels={'x': 'COMPANY','y': 'REVIEWS'}, title='Companies with highest 5 star reviews',color = company)
    st.plotly_chart(fig)


    # ### Model Selection and Training

    # In[7]:


    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from scipy.special import softmax


    # In[8]:


    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)


    # In[9]:

    @st.cache
    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'Negative' : scores[0],
            'Neutral' : scores[1],
            'Positive' : scores[2]
        }
        return scores_dict


    # ### WordCloud Function

    # In[10]:


    from wordcloud import WordCloud
    
    @st.cache
    def wc(data,bgcolor,title):
        plt.figure(figsize = (16,16))
        wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 40)
        wc.generate(' '.join(data))
        plt.imshow(wc)
        plt.axis('off')


    # ### Bigram Function

    # In[11]:

    #stopwords = stopwords.words('english')
    stoplist = nltk.corpus.stopwords.words('english') + ['though']
    @st.cache
    def bigram(reviews):
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
        ngrams = c_vec.fit_transform(reviews)
        count_values = ngrams.toarray().sum(axis=0)
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
        return df_ngram


    # ### Analysis on Positive Reviews

    # In[12]:


    res_roberta = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            pros = row['Pros']
            id= row['ID']
            roberta_result =polarity_scores_roberta(pros)
            res_roberta[id] = roberta_result

        except RuntimeError:
            print(f'RE broke for id:{id}')

        except ValueError:
            print(f'VE broke for id:{id}')


    # In[13]:


    results_pros = pd.DataFrame(res_roberta).T
    results_pros = results_pros.reset_index().rename(columns={'index': 'ID'})
    results_pros = results_pros.merge(df, how='left')


  
# In[82]:


    #user input for companies
    company_name = st.selectbox('Sentiment Analysis +ve',('pwc','accenture', 'deloitte','tcs','ey','kpmg'))
    if company_name == 'pwc':
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='PwC')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    elif(company_name == 'accenture'):
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='Accenture')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    elif(company_name == 'deloitte'):
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='Deloitte')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    elif(company_name == 'tcs'):
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='TCS')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    elif(company_name == 'ey'):
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='EY')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    elif(company_name == 'kpmg'):
        pros_df =results_pros[(results_pros['Positive'] > 0.95) & (results_pros['Company'] =='KPMG')]
        big =bigram(pros_df['Pros'])[:10]
        print(wc(pros_df['Pros'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)



    # ### Companies having Most Positive reviews

    # In[81]:


    pos_ratings=results_pros['Company'][results_pros['Positive'] >0.95].value_counts()
    fig = px.bar(pos_ratings,labels={'index': 'COMPANY','value': 'REVIEWS'},title='Companies with highest positive reviews',color =company)
    st.plotly_chart(fig)


    # ### Analysis on Negative Reviews

    # In[21]:


    res_roberta = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            cons = row['Cons']
            id= row['ID']
            roberta_result =polarity_scores_roberta(cons)
            res_roberta[id] = roberta_result

        except RuntimeError:
            print(f'RE broke for id:{id}')

        except ValueError:
            print(f'VE broke for id:{id}')


    # In[23]:


    results_cons = pd.DataFrame(res_roberta).T
    results_cons = results_cons.reset_index().rename(columns={'index': 'ID'})
    results_cons = results_cons.merge(df, how='left')


    # In[74]:


    company_name = st.selectbox('Sentiment Analysis -ve',('pwc','accenture', 'deloitte','tcs','ey','kpmg'))
    if company_name == 'pwc':
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='PwC')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)

    elif(company_name == 'accenture'):
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='Accenture')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)

    elif(company_name == 'deloitte'):
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='Deloitte')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)

    elif(company_name == 'tcs'):
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='TCS')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)

    elif(company_name == 'ey'):
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='EY')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)

    elif(company_name == 'kpmg'):
        cons_df =results_pros[(results_cons['Negative'] > 0.9) & (results_cons['Company'] =='KPMG')]
        big =bigram(cons_df['Cons'])[:10]
        print(wc(cons_df['Cons'],'black','Common Words'))
        fig = px.bar(big, x='bigram/trigram', y='frequency',color = 'bigram/trigram')
        st.plotly_chart(fig)


    # ### Companies having most Negative reviews

    # In[66]:


    neg_ratings=results_cons['Company'][results_cons['Negative'] >0.9].value_counts()
    fig = px.bar(neg_ratings,labels={'index': 'COMPANY','value': 'REVIEWS'},title='Companies with highest negative reviews',color =company)
    st.plotly_chart(fig)

##############    
elif option == 'Topic Modeling':
    st.subheader("Topic Modeling") 
    
    st.write("Every day, businesses deal with large volumes of unstructured text. From customer interactions in emails to online feedback and reviews. To deal with this large amount of text, we look towards topic modeling. A technique to automatically extract meaning from documents by identifying recurrent topics.Web scrap the Twitter data through web scraping tool/technique and  find out the way to categorise theme of the News that can be extracted using AI/ML technique aka Topic Modelling.")
    
    col1, col2 = st.columns([1, 1])
    with col1: 
        st.image("istockphoto-1291779592-170667a.jpg") 
    with col2: 
        st.write("""Deriving meaningful information from text data is the main objective. The purpose of Web Scraping will be focusing on creating a corpus of Simple Twitter topics. BERTopic has been used that combines transformer embeddings and clustering model algorithms to identify topics in NLP.
""")

    #Picking what NLP task you want to do
    option = st.selectbox('Topic Model',('game','technology', 'war','power','robotics')) #option is stored in this variable

    #upload file
    df=pd.read_csv('Technology - BBC News.csv')
    df1=df.copy()
    data = df1.drop(columns=['Title', 'Title_URL','Image','Time','Name','Description'], axis=1)

    # Print out the first rows of papers
    mask = data['Summary'].str.len() >=1
    reviews= data.loc[mask]
    reviews['Summary'] = reviews['Summary'].astype('str')
    docs = reviews.Summary.to_list()   
    
    
    BerTopic_model1 = BERTopic(verbose=True,embedding_model='xlm-r-bert-base-nli-stsb-mean-tokens', calculate_probabilities=True)
    #headline_topics, _ = model.fit_transform(docs)
    topics1, probabilities1=BerTopic_model1.fit_transform(docs)

    #@st.cache
    BerTopic_model2 = BERTopic(verbose=True,embedding_model='paraphrase-MiniLM-L3-v2', calculate_probabilities=True)
    #headline_topics, _ = model.fit_transform(docs)
    topics2, probabilities2=BerTopic_model2.fit_transform(docs)

    #@st.cache
    BerTopic_model3 = BERTopic(verbose=True,embedding_model='all-mpnet-base-v2', calculate_probabilities=True)
    #headline_topics, _ = model.fit_transform(docs)
    topics3, probabilities3=BerTopic_model3.fit_transform(docs)
    
    #@st.cache
    BerTopic_model4 = BERTopic(verbose=True,embedding_model='distilbert-base-nli-mean-tokens', calculate_probabilities=True)
    #headline_topics, _ = model.fit_transform(docs)
    topics4, probabilities4=BerTopic_model4.fit_transform(docs)

        


    if option == 'game':

        # Select most 10 similar topics for model1
        similar_topics1, similarity1 = BerTopic_model1.find_topics("game", top_n = 10)
        most_similar1 = similar_topics1[0]

        # Select most 10 similar topics for model2
        similar_topics2, similarity2 = BerTopic_model2.find_topics("game", top_n = 10)
        most_similar2 = similar_topics2[0]

        # Select most 10 similar topics for model3
        similar_topics3, similarity3 = BerTopic_model3.find_topics("game", top_n = 10)
        most_similar3 = similar_topics3[0]

        # Select most 10 similar topics for model4
        similar_topics4, similarity4 = BerTopic_model4.find_topics("game", top_n = 10)
        most_similar4 = similar_topics4[0]

    elif option == 'technology':
        # Select most 10 similar topics for model1
        similar_topics1, similarity1 = BerTopic_model1.find_topics("technology", top_n = 10)
        most_similar1 = similar_topics1[0]

        # Select most 10 similar topics for model2
        similar_topics2, similarity2 = BerTopic_model2.find_topics("technology", top_n = 10)
        most_similar2 = similar_topics2[0]

        # Select most 10 similar topics for model3
        similar_topics3, similarity3 = BerTopic_model3.find_topics("technology", top_n = 10)
        most_similar3 = similar_topics3[0]

        # Select most 10 similar topics for model4
        similar_topics4, similarity4 = BerTopic_model4.find_topics("technology", top_n = 10)
        most_similar4 = similar_topics4[0]


    elif option == 'war':
        # Select most 10 similar topics for model1
        similar_topics1, similarity1 = BerTopic_model1.find_topics("war", top_n = 10)
        most_similar1 = similar_topics1[0]

        # Select most 10 similar topics for model2
        similar_topics2, similarity2 = BerTopic_model2.find_topics("war", top_n = 10)
        most_similar2 = similar_topics2[0]

        # Select most 10 similar topics for model3
        similar_topics3, similarity3 = BerTopic_model3.find_topics("war", top_n = 10)
        most_similar3 = similar_topics3[0]

        # Select most 10 similar topics for model4
        similar_topics4, similarity4 = BerTopic_model4.find_topics("war", top_n = 10)
        most_similar4 = similar_topics4[0]

    elif option == 'power':
        # Select most 10 similar topics for model1
        similar_topics1, similarity1 = BerTopic_model1.find_topics("power", top_n = 10)
        most_similar1 = similar_topics1[0]

        # Select most 10 similar topics for model2
        similar_topics2, similarity2 = BerTopic_model2.find_topics("power", top_n = 10)
        most_similar2 = similar_topics2[0]

        # Select most 10 similar topics for model3
        similar_topics3, similarity3 = BerTopic_model3.find_topics("power", top_n = 10)
        most_similar3 = similar_topics3[0]

        # Select most 10 similar topics for model4
        similar_topics4, similarity4 = BerTopic_model4.find_topics("power", top_n = 10)
        most_similar4 = similar_topics4[0]


    elif option == 'robotics':
        # Select most 10 similar topics for model1
        similar_topics1, similarity1 = BerTopic_model1.find_topics("robotics", top_n = 10)
        most_similar1 = similar_topics1[0]

        # Select most 10 similar topics for model2
        similar_topics2, similarity2 = BerTopic_model2.find_topics("robotics", top_n = 10)
        most_similar2 = similar_topics2[0]

        # Select most 10 similar topics for model3
        similar_topics3, similarity3 = BerTopic_model3.find_topics("robotics", top_n = 10)
        most_similar3 = similar_topics3[0]

        # Select most 10 similar topics for model4
        similar_topics4, similarity4 = BerTopic_model4.find_topics("robotics", top_n = 10)
        most_similar4 = similar_topics4[0]

    st.write("HuggingFace Transformer Model: 'xlm-r-bert-base-nli-stsb-mean-tokens'")

    st.write(similarity1[0])

    st.write("HuggingFace Transformer Model: 'paraphrase-MiniLM-L3-v2'")

    st.write(similarity2[0])

    st.write("HuggingFace Transformer Model: 'all-mpnet-base-v2'")

    st.write(similarity3[0])

    st.write("HuggingFace Transformer Model: 'distilbert-base-nli-mean-tokens'")

    st.write(similarity4[0])

    maximum = max(similarity1[0], similarity2[0], similarity3[0], similarity4[0])

    if similarity1[0]==maximum :
        best_model=BerTopic_model1
        st.subheader("Best Performing HuggingFace Transformer Model")  
        st.write("xlm-r-bert-base-nli-stsb-mean-tokens") 
    elif similarity2[0]==maximum :
        best_model=BerTopic_model2
        st.subheader("Best Performing HuggingFace Transformer Model")  
        st.write("paraphrase-MiniLM-L3-v2")
    elif similarity3[0]==maximum :
        best_model=BerTopic_model3
        st.subheader("Best Performing HuggingFace Transformer Model")  
        st.write("all-mpnet-base-v2")
    elif similarity4[0]==maximum :
        best_model=BerTopic_model4
        st.subheader("Best Performing HuggingFace Transformer Model")  
        st.write("distilbert-base-nli-mean-tokens")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file,nrows=500)
        mask = data['text'].str.len() >=1
        reviews= data.loc[mask]
        reviews['text'] = reviews['text'].astype('str')
        new_docs = reviews.text.to_list()    
        topics, probs =best_model.fit_transform(new_docs)

            #st.write(new_docs)

            #st.write(best_model.visualize_barchart(top_n_topics=20, n_words=20))
            #st.plotly_chart(best_model.visualize_topics())
            #fig1=best_model.visualize_distribution(probs[0])
            #fig2=best_model.visualize_term_rank()
            #fig3=best_model.visualize_hierarchy()
            #fig4=best_model.get_topic_info()

            #fig5=best_model.visualize_topics()
            #st.plotly_chart(fig5)

        fig1=best_model.visualize_topics()
        st.plotly_chart(fig1)

        fig2=best_model.visualize_heatmap()
        st.plotly_chart(fig2)

        fig3=best_model.visualize_documents(new_docs)
        st.plotly_chart(fig3)

        fig4=best_model.visualize_term_rank()
        st.plotly_chart(fig4)

        fig5=best_model.visualize_barchart(top_n_topics=10)
        st.plotly_chart(fig5,use_container_width=True) 
