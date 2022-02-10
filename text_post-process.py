import re


def rep_search(text):
    ''''
    Function to catch repeating phrases
    '''
     
    for x in range(len(text)):
        
        exp = rf'((\(*[€«»]*\s*[A-Z]\S+\b\s*[€]*\s*[«»]*\)*)(\s+\(*[€«»]*\s*\b\S+\b\s*[€]*\s*[«»]*\)*){{{x}}}(\s+\(*[€«»]*\s*\b\S+\s*[€]*\s*[«»]*\)*[\.]?))(\1)+'
        search = re.search(exp, text, re.UNICODE)
        find = re.findall(exp, text, re.UNICODE)
        
        if search:
            # starting index of the phrase repetitions
            start = search.span()[0]
            # area containing single instance of phrase that is repeated
            middle = start + len(find[0][0])
            # ending index of the phrase repetitions
            end = search.span()[1]
            
            if start == 0:
                # If the repeated phrase is at the beginning of the string
                new_text = text[start:middle] + text[end:]
                return rep_search(new_text)
            
            else:
                new_text = text[0:middle] + text[end:len(text)]
                return rep_search(new_text)
            
    for x in range(len(text)):
        
        exp2 = rf'(\(*[€«»]*\s*\b\S+\b\s*[€]*\s*[«»]*\)*(\s+\(*[€«»]*\s*\b\S+\b\s*[€]*\s*[«»]*\)*[\.]?){{{x}}})\s*(\1(\s*|\.))+'
        search2 = re.search(exp2, text, re.UNICODE)
        find2 = re.findall(exp2, text, re.UNICODE)
        
        if search2:
            # starting index of the phrase repetitions
            start = search2.span()[0]
            # area containing single instance of phrase that is repeated
            middle = start + len(find2[0][0])
            # ending index of the phrase repetitions
            end = search2.span()[1]
           
            if start == 0:
               # If the repeated phrase is at the beginning of the string
               new_text = text[start:middle] + ' ' + text[end:]
               return rep_search(new_text)
    
            else:
               new_text = text[0:middle] + ' ' + text[end:len(text)]
               return rep_search(new_text) 
           
    if not search and not search2:
    # If there are no repetitions in the string
        return text
   
