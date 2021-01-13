# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:39:32 2020

@author: Roxy8
"""

from PitchToFrets import getPossibleTuples
from useModelsToConvert import use_rnns




def getTabFromFile(file):
    
    #tupples = use_rnns(file)
    tupples = use_rnns("../Tab/Example_Tab/example_1.mid")
    
    tab = ["E|","B|","G|","D|","A|","E|"]
    
    finalTab = "Title: Unknown \nAuthor: Unknown \nDate: Unknown \n \n"
    
    nb_notes = 0
    nb_mesures = 0
    
    for tupple in tupples:
        
        if(nb_notes==5):
            nb_notes = 0
            nb_mesures+=1
            for i in range(len(tab)):
                tab[i]+="|"
        
        if(nb_mesures == 5):
            for line in tab:
                finalTab+=line
                finalTab+="\n"   
            finalTab+="\n" 
            tab = ["E|","B|","G|","D|","A|","E|"]
            nb_mesures = 0
        
        for i in range(len(tab)):
            if(tupple[0]==6-i):
                tab[i]+="-"+str(tupple[1])
                if(tupple[1]<10):
                    tab[i]+="-"
            else:
                tab[i]+="---"
            
            
        nb_notes+=1
        
                
                
    for line in tab:
        finalTab+=line
        finalTab+="|\n" 
                
    
    if __name__ == "__main__":
        print(finalTab)
        
    return finalTab

if __name__ == "__main__":
    # Testing the function
    input_file = "../Tab/Example_Tab/example_1.mid"
    getTabFromFile(input_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    