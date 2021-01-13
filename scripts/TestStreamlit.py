# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:09:06 2020

@author: Roxy8
"""

import streamlit as st
from PitchToFrets import getPossibleTuples
from FretToPitch import getAssociatedNote
from music21 import converter, midi
import os
from PitchesToTab import getTabFromFile
import base64
import pandas as pd
import time
import streamlit as st
import SessionState
#from st.script_runner import RerunException




def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))


session_state = SessionState.get(page=0, file=0, tab=0)  # Pick some initial values.


st.title("MIDI to Tab")
st.text(
"""This wonderful app will allow you to convert your MIDI file into guitar tablatures.
These tablatures will have a very good quality!""")



if session_state.page == 0:
    
    st.header("First step: select the MIDI file you want to convert.")
    
    
    FILE_TYPES = ["mid", "midi"]
    file = st.file_uploader("Upload file", type=FILE_TYPES)
    
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
    else:
        show_file.info("Please wait, the application is loading...")
        file_content=file.getvalue()
        
        pitches = converter.parse(file_content).pitches
        st.text('You selected a file. \n Here are the 10 first notes:')
        notes = ""
        for i in range(0,9):
            notes=notes+str(pitches[i])+", "
        notes=notes+str(pitches[10])
        st.text(notes)
        show_file.empty()
        
        if st.button("See the tablature"):
            session_state.file = file
            session_state.page = 1
            rerun()
        
        
elif session_state.page == 1:
    
    st.header("Second step: see the tablature and choose the way you want to get it.")
        
    file_name = session_state.file.name.split(".")[0]
    session_state.tab = getTabFromFile(session_state.file.getvalue(), file_name)
    
    st.write('This is the begining of the tab file:')
    st.text(session_state.tab[:1000])
    
    if st.button('Download this new tablature'):
        session_state.page = 2
        rerun()
        
    if st.button('Get this tablature by email'):
        email = st.text_input("Write your email", "example@gmail.com")
        st.write("(This functionality is not implemented yet)")
        
        
elif session_state.page == 2:
    
    st.header("Third step: get your wonderful tablature!")
    
    st.write("Congratulations, your tablature is ready to be downloaded!")
    
    file_name = session_state.file.name.split(".")[0]
    
    tmp_download_link = download_link(session_state.tab, file_name+".tab", 'Click here to download it!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
    
    if st.button("Create a new one!"):
        session_state.page = 0
        rerun()










    
 
    
        
    