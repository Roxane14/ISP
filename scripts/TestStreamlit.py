# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:09:06 2020

@author: Roxy8
"""

import streamlit as st
from PitchToFrets import getPossibleTuples
from FretToPitch import getAssociatedNote
from music21 import converter
import os
from PitchesToTab import getTabFromPitches
import base64
import pandas as pd



st.title("MIDI to Tab")
st.text(
"""This wonderful app will allow you to convert your MIDI file into guitar tablatures.
Unfortunately, for the moment, it is not finished.
But no worries, you can still try some functionalities!""")

st.header("First experiment: upload MIDI file and get a tablature")
st.subheader("This part doesn't work in reality. But it's a kind of mock-up.")

def file_selector(folder_path=''):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a MIDI file', filenames)
    if selected_filename.lower().endswith('.mid'):
        return os.path.join(folder_path, selected_filename)
    else:
        return 0
    
    
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    

filename = file_selector()

if(filename!=0):
    st.write('You selected `%s`. \n Here are the 10 first notes:' % filename)
    pitches = converter.parse(filename).pitches
    notes = ""
    for i in range(0,9):
        notes=notes+str(pitches[i])+", "
    notes=notes+str(pitches[10])
    st.write(notes)
    tab = getTabFromPitches(pitches)
    st.write('This is the begining of the tab file:')
    st.write(tab[:1000])
    if st.button('Download this new tablature'):
        tmp_download_link = download_link(tab, 'YOUR_INPUT.tab', 'Click here to download your tab!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
else:
    st.write('You need to select a MIDI file')



st.header("Another experiment: convert a note into possible finger position")
st.subheader("This is really a beautiful part")
input = st.text_input("Write a note", "G4")
output = getPossibleTuples(input)
st.write(output)

st.header("Last experiment: convert finger position into the corresponding note")
st.subheader("This is also a nice part")
inputFret = int(st.text_input("Write the fret", "12"))
inputString = int(st.text_input("Write the string", "3"))
output = (getAssociatedNote(inputFret, inputString))
st.write(output)