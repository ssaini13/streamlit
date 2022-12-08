#Necessary imports
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from bertopic import BERTopic
import pandas as pd



#Headings for Web Application
st.sidebar.title("NLP Web App:")
st.sidebar.subheader("BertTopic")

#Picking what NLP task you want to do
option = st.sidebar.selectbox('Topic Model',('game','technology', 'war','power','robotics')) #option is stored in this variable

#upload file

BerTopic_model1 = BERTopic.load("model_xlm-r-bert-base-nli-stsb-mean-tokens")
BerTopic_model2 = BERTopic.load("model_paraphrase-MiniLM-L3-v2")
BerTopic_model3 = BERTopic.load("model_all-mpnet-base-v2")
BerTopic_model4 = BERTopic.load("model_distilbert-base-nli-mean-tokens")        


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
                        
    
    
    
    