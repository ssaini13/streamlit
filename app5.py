#Necessary imports
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from bertopic import BERTopic


import pandas as pd



#Headings for Web Application
st.sidebar.title("NLP Web App:")
st.sidebar.subheader("BertTopic")

#Picking what NLP task you want to do
option = st.sidebar.selectbox('Transformer Model',('xlm-r-bert-base-nli-stsb-mean-tokens', 'paraphrase-MiniLM-L3-v2')) #option is stored in this variable

#upload file
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df1=df.copy()
        data = df1.drop(columns=['Title', 'Title_URL','Image','Time','Name','Description'], axis=1)

        # Print out the first rows of papers
        mask = data['Summary'].str.len() >=1
        reviews= data.loc[mask]
        reviews['Summary'] = reviews['Summary'].astype('str')
        docs = reviews.Summary.to_list()
        
        if option == 'xlm-r-bert-base-nli-stsb-mean-tokens':
   
        
            model1 = BERTopic(verbose=True,embedding_model='xlm-r-bert-base-nli-stsb-mean-tokens',                                                                                         calculate_probabilities=True)
            #headline_topics, _ = model.fit_transform(docs)
            topics, probabilities=model1.fit_transform(docs)
        
            # Select most 10 similar topics for model1
            similar_topics1, similarity1 = model1.find_topics("game", top_n = 10)
            most_similar1 = similar_topics1[0]



            st.header("Results")
            #st.write(topic_words)
            st.write("Selected Topic: Game")
            
            
            fig1=model1.visualize_topics()
            st.plotly_chart(fig1)
            
            fig2=model1.visualize_heatmap()
            st.plotly_chart(fig2)
            
            fig3=model1.visualize_documents(docs)
            st.plotly_chart(fig3)
            
            fig4=model1.visualize_term_rank()
            st.plotly_chart(fig4)
            
            fig5=model1.visualize_barchart(top_n_topics=20)
            st.plotly_chart(fig5,use_container_width=True)

            
   
        elif option == 'paraphrase-MiniLM-L3-v2':
            pass
        
