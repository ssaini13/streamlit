#Necessary imports
import streamlit as st
import numpy as np 
import pandas as pd 

from wordcloud import WordCloud
from tqdm.notebook import tqdm
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import base64
from PIL import Image
from bertopic import BERTopic

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
      pass
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