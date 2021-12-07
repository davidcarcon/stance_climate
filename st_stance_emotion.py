import streamlit as st
import modelos as md
from scipy.special import softmax
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from googletrans import Translator
nltk.download('vader_lexicon')

translator = Translator()
sia = SentimentIntensityAnalyzer()

tok_emo, model_emo, labels_emo = md.return_models_labels('emotion')
tok_stance, model_stance, labels_stance = md.return_models_labels('stance-climate')

texto = st.text_input('Escribe el texto del tweet')
text_mod = md.preprocess(texto)

if texto != '':
        st.markdown(f"""
        ### Texto:
        {text_mod}
        """)
        text_en = translator.translate(text_mod, dest='en').text
        st.markdown("""### Emociones """)
        encoded_input = tok_emo(text_en, return_tensors='tf')
        output = model_emo(encoded_input)
        scores = output[0][0].numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
                l = labels_emo[ranking[i]]
                s = scores[ranking[i]]
                st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
        #####
        st.markdown(""" ### Postura Cambio Climático """)
        encoded_inputt = tok_stance(text_en, return_tensors='tf')
        outputt = model_stance(encoded_inputt)
        scorest = outputt[0][0].numpy()
        scorest = softmax(scorest)
        rankingt = np.argsort(scorest)
        rankingt = rankingt[::-1]
        for i in range(scorest.shape[0]):
                l = labels_stance[rankingt[i]]
                s = scores[rankingt[i]]
                st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
        st.markdown(""" ### VADER """)
        st.write(sia.polarity_scores(text_en))
