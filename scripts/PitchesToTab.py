# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:39:32 2020

@author: Roxy8
"""

from PitchToFrets import getPossibleTuples
#from useModelsToConvert import use_rnns

def use_rnns(file):
    return [(4, 7), (6, 5), (3, 17), (6, 0), (4, 7), (5, 1), (3, 7), (5, 3), (5, 1), (3, 7), (3, 5), (3, 3), (4, 12), (6, 8), (6, 8), (6, 8), (6, 8), (5, 10), (5, 5), (5, 10), (6, 8), (5, 15), (6, 8), (5, 10), (5, 8), (5, 6), (5, 8), (5, 10), (4, 7), (5, 10), (4, 12), (4, 9), (4, 7), (5, 1), (3, 7), (4, 7), (5, 1), (3, 7), (3, 5), (3, 2), (3, 5), (3, 7), (3, 5), (3, 7), (5, 1), (3, 7), (5, 1), (3, 5), (3, 7), (3, 5), (6, 5), (6, 3), (6, 0), (4, 7), (5, 1), (3, 7), (5, 1), (5, 3), (5, 1), (4, 2), (3, 5), (3, 7), (3, 5), (3, 3), (3, 5), (3, 7), (3, 5), (3, 7), (5, 1), (6, 1), (4, 7), (5, 1), (3, 7), (5, 1), (4, 7), (6, 0), (6, 3), (6, 0), (6, 3), (4, 14), (6, 8), (5, 17), (5, 15), (5, 15), (6, 8), (5, 15), (6, 8), (6, 8), (6, 15), (6, 15), (6, 13), (6, 10), (6, 10), (6, 8), (5, 10), (5, 13), (6, 10), (6, 8), (5, 10), (6, 8), (6, 10), (6, 8), (5, 10), (6, 8), (6, 10), (5, 13), (5, 10), (6, 8), (6, 10), (6, 8), (5, 10), (6, 8), (6, 10), (5, 13), (5, 10), (6, 8), (6, 10), (6, 8), (5, 10), (6, 8), (6, 10), (5, 13), (5, 10), (6, 8), (6, 10), (6, 8), (5, 10), (6, 8), (6, 10), (5, 13), (5, 10), (6, 8), (6, 10), (6, 8), (5, 10), (6, 8), (6, 15), (5, 17), (6, 15), (5, 17), (6, 10), (5, 10), (6, 8), (5, 10), (5, 8), (5, 5), (5, 3), (4, 5), (6, 0), (5, 3), (4, 2), (4, 5), (5, 5), (5, 8), (5, 3), (2, 10), (5, 3), (4, 5), (5, 3), (4, 0), (4, 2), (4, 5), (5, 3), (4, 5), (4, 2), (2, 10), (4, 2), (4, 5), (5, 3), (5, 3), (4, 5), (6, 0), (5, 10), (6, 8), (6, 10), (5, 8), (5, 10), (6, 10), (6, 8), (6, 10), (5, 8), (6, 10), (5, 8), (6, 0), (5, 8), (6, 10), (5, 8), (5, 5), (5, 8), (5, 15), (5, 8), (5, 5), (5, 8), (6, 8), (5, 10), (5, 8), (5, 15), (5, 8), (5, 15), (6, 8), (5, 10), (5, 8), (6, 8), (5, 15), (5, 8), (5, 15), (6, 8), (5, 10), (5, 8), (6, 8), (5, 10), (5, 8), (5, 10), (5, 8), (6, 0), (5, 3), (4, 5), (5, 3), (5, 5), (5, 3), (4, 5), (4, 2), (2, 10), (2, 8), (4, 0), (4, 2), (4, 5), (2, 10), (3, 7), (4, 7), (3, 5), (3, 7), (4, 7), (5, 1), (4, 7), (5, 5), (6, 3), (4, 14), (5, 15), (5, 15), (6, 5), (5, 15), (5, 15), (6, 5), (5, 15), (6, 5), (5, 15), (6, 11), (5, 17), (6, 11), (6, 11), (5, 14), (6, 4), (5, 7), (6, 7), (5, 10), (4, 12), (5, 7), (4, 9), (5, 3), (5, 1), (6, 0), (5, 7), (5, 8), (5, 3), (5, 8), (5, 7), (4, 9), (5, 3), (5, 1), (5, 3), (6, 0), (6, 3), (5, 10), (6, 8), (5, 14), (5, 17), (6, 11), (5, 14), (5, 17), (6, 11), (6, 9), (5, 17), (5, 17), (6, 13), (6, 14), (6, 15), (6, 17), (5, 17), (6, 17), (5, 17), (6, 20), (6, 17), (6, 20), (6, 17), (6, 20), (5, 17), (6, 20), (6, 20), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (5, 17), (6, 20), (6, 17), (6, 10), (6, 20)]



def getTabFromFile(file):
    
    tupples = use_rnns(file)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    