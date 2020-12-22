#function extractTuples
#input : file
#output : all the tuples(string, fret) of the file

def getSingleVec(vecset, i):
    return[row[i] for row in vecset]

def extractTuples(file):
    #file = open('D:\\Documents\\Ensim\\S9b\\ISP\\tg tab\\tab_test.tab')
    Lines = file.readlines()
                
    standard_tuning = list("EBGDAE")
    tuples = [] #(string, fret)
    vecset = list("EBGDAE")
    
    for line in range(len(Lines)-6):
        vec = []
        for i in range(6):
            vec.append(Lines[line + i])
            vec[i] = vec[i][0]
            if (vec == standard_tuning):
                for i in range(line,line+6):
                    vecset[i-line] += Lines[i]
                    
    vecset2 = [list(vecset[0]),
               list(vecset[1]),
               list(vecset[2]),
               list(vecset[3]),
               list(vecset[4]),
               list(vecset[5])]
    
    i = 0
    while (i < len(vecset2[0]) - 1):
        vec1 = getSingleVec(vecset2, i)
        vec2 = getSingleVec(vecset2, i+1)
        if (vec1.count('-') == 5):
            for element in range(6):
                if (vec1[element].isnumeric()):
                    if (vec2[element].isnumeric()):
                        if int(vec1[element]+vec2[element]) <= 24:
                            tuples.append([6-element, int(vec1[element]+vec2[element])])
                        i += 1
                    else:
                        tuples.append([6-element, int(vec1[element])])
        i += 1
        
    return tuples
