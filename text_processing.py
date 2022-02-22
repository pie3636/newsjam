import re


class Post:
    '''
    Class for text post-processing functions
    - to be run after the summary is generated
    '''
    
    def __init__(self):
        pass
    
    def rep_search(self, text):
        
        '''
        Function to catch and get rid of repeating phrases
        '''
         
        for x in range(len(text.split(' '))):
            
            exp = rf'((\(*[\$€£«\'\"]*\s*[A-Z]*\S+\b\s*[\$€£\.\,\!\?\&\%\s]*[»\'\"]?\)*)([\s\,\:\;]+\(*[\$€£«\'\"]*\s*\b\S+\b\s*[\$€£\.\,\!\?\&\%\s]*[»\'\"]?\)*){{{x}}}([\s\,\:\;]+\(*[\$€£«\'\"]*\s*\b\S+\s*[\$€£\.\,\!\?\&\%\s]*[»\'\"]?\)*[\.\?\!]?))\s*(\1)+'
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
                return self.rep_search(new_text)
                
        if not search:
            # If there are no repetitions in the string
            return text
 


class Pre:
    '''
    Class for text pre-processing functions
    - to be run on articles before summary is generated
    '''
    
    def __init__(self):
        pass
    
    def fr_phrases(self, text):
        
        '''
        Function to get rid of unnecessary phrases found in some of our French corpus articles
        '''
        
        exp1 = "Ce contenu est bloqué car vous n\'avez pas accepté les traceurs. "
        exp2 = "En cliquant sur « J’accepte », les traceurs seront déposés et vous pourrez visualiser les contenus . "
        exp3 = "En cliquant sur « J’accepte tous les traceurs », vous autorisez des dépôts de traceurs pour le stockage de vos données sur nos sites et applications à des fins de personnalisation et de ciblage publicitaire. "
        
        exp_list = [exp1, exp2, exp3]
        
        for exp in exp_list:
            search = re.search(exp, text)
        
            if search:
                new_text = re.sub(exp, '', text)
                return self.fr_phrases(new_text)
                
        if not search:
            return text
        
        
    def en_phrases(self, text):
        '''
        Function to get rid of unnecessary phrases found in some of our English corpus articles
        '''
        
        exp1 = "Sign up to the daily Business Today email or follow Guardian Business on Twitter at @BusinessDesk "
        exp2 = "Sign up to receive the top stories from Guardian Australia every morning "
        exp3 = "Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhones or the Google Play store on Android phones by searching for 'The Guardian'. "
        exp4 = "If you already have the Guardian app, make sure you’re on the most recent version. "
        exp5 = "In the Guardian app, tap the yellow button at the bottom right, then go to Settings (the gear icon), then Notifications. "
        exp6 = "Turn on sport notifications. Was this helpful? Thank you for your feedback. "
        exp7 = "Allow TikTok content? This article includes content provided by TikTok . We ask for your permission before anything is loaded, as they may be using cookies and other technologies. To view this content, click 'Allow and continue'. Allow and continue "
        exp8 = "Sign up to receive the top stories from Guardian Australia every morning "
        exp9 = "Sign up to receive an email with the top stories from Guardian Australia every morning "
        
        exp_list = [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9]
        
        for exp in exp_list:
            search = re.search(exp, text)
            
            if search:
                new_text = re.sub(exp, '', text)
                return self.en_phrases(new_text)
            
        if not search:
            return text
  
