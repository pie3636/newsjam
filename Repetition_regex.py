import re

test1 = 'This is a test.This is a test.This is a test.This is a test.This is a test.This is a test.'
test2 = 'Yesterday I went to the store and now I am going home. Yesterday I went to the store and now I am going home. Yesterday I went to the store and now I am going home.'
test3 = 'This iS a Test. This iS a Test. This iS a Test. This iS a Test. This iS a Test.'

def rep_search(text):

    find = re.findall(r'(\b[A-Z][{a-z}\s\.]+)(\1)+', text)
    if find:
        new_text = text[0:len(find[0][0])]
        return print(new_text)

    for x in range(len(text)):
        exp1 = rf'(\b\S+\b(\s+\b\S+\b[\.]?){{{x}}})\s*(\1(\s*|\.))+'
        find2 = re.findall(exp1, text)

        exp2 = rf'(([A-Z]\S+\b)(\s+\b\S+\b){{{x}}}(\s+\b\S+[\.]?))(\1)+'
        find3 = re.findall(exp2, text)

        if find2:
            new_text = text[0:len(find2[0][0])]
            return print(new_text)
            break


        elif find3:
            new_text = text[0:len(find3[0][0])]
            return print(new_text)
            break


rep_search(test2)