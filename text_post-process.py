import re



def rep_search(text):
    
    ''''
    Function to catch repeating phrases
    '''
     
    for x in range(len(text.split(' '))):
        
        exp = rf'((\(*[€«]*\s*[A-Z]\S+\b\s*[€\.\,\&\s]*[»]?\)*)([\s\,\:]+\(*[€«]*\s*\b\S+\b\s*[€\.\,\&\s]*[»]?\)*){{{x}}}([\s\,\:]+\(*[€«]*\s*\b\S+\s*[€\.\,\&\s]*[»]?\)*[\.\?\!]?))\s*(\1)+'
        search = re.search(exp, text, re.UNICODE)
        find = re.findall(exp, text, re.UNICODE)
        
        if search:
            # starting index of the phrase repetitions
            start = search.span()[0]
            # area containing single instance of phrase that is repeated
            middle = start + len(find[0][0])
            # ending index of the phrase repetitions
            end = search.span()[1]
            
            new_text = text[0:middle] + text[end:]
            return rep_search(new_text)
            
        
        exp2 = rf'(\(*[€«]*\s*\b\S+\b\s*[€\.\,\!\?\&\s]*[»]?\)*([\s\,\:]+\(*[€«]*\s*\b\S+\b\s*[€\.\,\!\?\&\s]*[»]?\)*){{{x}}}[\.\?\!]?)\s*(\1)+'
        search2 = re.search(exp2, text, re.UNICODE)
        find2 = re.findall(exp2, text, re.UNICODE)
        
        if search2:
            # starting index of the phrase repetitions
            start = search2.span()[0]
            # area containing single instance of phrase that is repeated
            middle = start + len(find2[0][0])
            # ending index of the phrase repetitions
            end = search2.span()[1]
           
            new_text = text[0:middle] + text[end:]
            return rep_search(new_text)
    
    if not search and not search2:
        # If there are no repetitions in the string
        return text
 


def blocked_phrase(text):
    
    exp1 = 'Ce contenu est bloqué car vous n\'avez pas accepté les traceurs. '
    exp2 = 'En cliquant sur « J’accepte », les traceurs seront déposés et vous pourrez visualiser les contenus . '
    exp3 = 'En cliquant sur « J’accepte tous les traceurs », vous autorisez des dépôts de traceurs pour le stockage de vos données sur nos sites et applications à des fins de personnalisation et de ciblage publicitaire. '
    
    
    search1 = re.search(exp1, text)
    
    if search1:
        start = search1.span()[0]
        end = search1.span()[1]
        
        if start == 0:
            # If the repeated phrase is at the beginning of the string
            new_text = text[end:]
            return blocked_phrase(new_text)
            
        else:
            new_text = text[0:start]+ text[end:len(text)]
            return blocked_phrase(new_text)
   
    
    search2 = re.search(exp2, text)
    
    if search2:
        start2 = search2.span()[0]
        end2 = search2.span()[1]
    
        if start2 == 0:
            # If the phrase is at the beginning of the string
            new_text = text[end2:]
            return blocked_phrase(new_text)
            
        else:
            new_text = text[0:start2]+ text[end2:len(text)]
            return blocked_phrase(new_text)
           
            
    search3 = re.search(exp3, text)
    
    if search3:
        start3 = search3.span()[0]
        end3 = search3.span()[1]
    
        if start3 == 0:
            # If the phrase is at the beginning of the string
            new_text = text[end3:]
            return blocked_phrase(new_text)
            
        else:
            new_text = text[0:start3]+ text[end3:len(text)]
            return blocked_phrase(new_text)
           
            
    if not search1 and not search2 and not search3:
        return text
   
