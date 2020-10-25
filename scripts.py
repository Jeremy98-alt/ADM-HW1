# Say "Hello, World!" With Python

print("Hello, World!")

# Python If-Else

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())

    if n % 2 != 0:
        print("Weird")
    elif n >= 2 and n <= 5:
        print("Not Weird")
    elif n >= 6 and n <= 20:
        print("Weird")
    else :
        print("Not Weird")

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)

# Loops 

if __name__ == '__main__':
    n = int(input())

    for i in range(0,n):
        if i < n:
            print(i**2)

# Write a function

import calendar

def is_leap(year):    
    return calendar.isleap(year)

year = int(input())
print(is_leap(year))

# Print Function

if __name__ == '__main__':
    n = int(input())

    l = ""
    for i in range(0,n):
        l += str(i+1)
    
    print(l)

# List Comprehensions 

#(I saw the solution code in terms of code write style)
if __name__ == '__main__':
    x, y, z, n = [int(input()) for _ in range(4)]

    print([ [x2,y2,z2] for x2 in range(0,x+1) 
                            for y2 in range(0,y+1) 
                                for z2 in range(0,z+1) 
                                    if x2 + z2 + y2 != n ])

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    arr = sorted(arr)
    
    max_value = arr[len(arr)-1]
    i = len(arr)-2
    while i >= 0:
        if max_value != arr[i]:
            print(arr[i])
            break
        i-=1
    
# Nested Lists

#(I saw the solution code in terms of code write style, in this half code part)
if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([score, name])

    students.sort()
    b = [i for i in students if i[0] != students[0][0]]
    c = [j for j in b if j[0] == b[0][0]]
    
    c.sort(key=lambda x: x[1])
    for i in range(len(c)):
        print(c[i][1])

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    query_scores = student_marks[query_name]
    avg = sum(query_scores)/len(query_scores)

    print("{0:.2f}".format(avg))

# Lists

if __name__ == '__main__':
    N = int(input())

    list = []
    for i in range(N):
        command = input().strip().split(" ")
        if command[0] == "insert":
            list.insert(int(command[1]), int(command[2]))

        if command[0] == "print":
            print(list)

        if command[0] == "append":
            list.append(int(command[1]))

        if command[0] == "remove":
            list.remove(int(command[1]))

        if command[0] == "pop":
            list.pop()

        if command[0] == "reverse":
            list.reverse()

        if command[0] == "sort":
            list.sort()

# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

    t = []
    for num in integer_list:
        t.append(num)
    
    print(hash(tuple(t)))

# sWAP cASE

def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String Split and Join

def split_and_join(line):
    line = line.split()
    line = "-".join(line)
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

def print_full_name(a, b):
    print(f"Hello {a} {b}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations

def mutate_string(string, position, character):
    s = list(string)
    s[position] = character
    string = ''.join(s)
    return string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a string

def count_substring(string, sub_string):
    count = 0
    i = string.find(sub_string)
    while i != -1:
        count += 1
        i = string.find(sub_string, i+1)

    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

# String Validators

if __name__ == '__main__':
    s = input()

    print(any(c.isalnum() for c in s))
    print(any(c.isalpha() for c in s))
    print(any(c.isdigit() for c in s))
    print(any(c.islower() for c in s))
    print(any(c.isupper() for c in s))

# Text Alignment 

#(I saw the code in terms logic way to put the correct distances)
thickness = int(input()) #This must be an odd number
c = 'H' #Replace all ______ with rjust, ljust or center.

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)) #Top Cone

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)) #Top Pillars

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6)) #Middle Belt

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)) #Bottom Pillars     

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)) #Bottom Cone

# Text Wrap

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string,max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

n, m = map(int, input().split())

path = ".|."
for i in range(1,n,2): 
    print((i * path).center(m, "-"))

print("WELCOME".center(m,"-"))

for i in range(n-2,-1,-2): 
    print((i * path).center(m, "-"))

# String Formatting

#(I saw the code in terms logic way to put the correct distances, in this case there was a "bug" to display the outcomes)
def print_formatted(number):
    # your code goes here
    width = len("{0:b}".format(number))
    for i in range(1,n+1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=width))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Alphabet Rangoli

#(I saw the code in terms logic way, in this case there was a "bug" to display the outcomes)
def print_rangoli(size):
    # your code goes here
    strAlph = 'abcdefghijklmnopqrstuvwxyz'[0:size]
    
    for i in range(size-1, -size, -1):
        print ("--"*abs(i)+ '-'.join(strAlph[size:abs(i):-1] + strAlph[abs(i):size])+"--"*abs(i))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!
import math
import os
import random
import re
import sys

def solve(s):
    inp = s.split(" ")
    
    wrdF = ""
    for w in inp:
        wrdF += w.capitalize() + " "

    return wrdF

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

# The Minion Game

def minion_game(string):
    vowels = "AEIOU"

    KScore = SScore = 0
    for i in range(len(string)):
        if string[i] in vowels:
            KScore += (len(string)-i)
        else:
            SScore += (len(string)-i)

    if KScore > SScore:
        print("Kevin", KScore)
    elif KScore < SScore:
        print("Stuart", SScore)
    else:
        print("Draw")
 
if __name__ == '__main__':
    s = input()
    minion_game(s)

# Merge the Tools!

def merge_the_tools(string, k):
    for x in range(0, len(string), k):
        subs = ""
        for y in string[x : x + k]:
            if y not in subs:
                subs += y          
        print(subs)
        
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# Introduction to Sets

def average(array):
    return sum(set(array))/len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# No Idea!

n, m = map(int,input().split())
arrayToBe = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))

happy = 0
for elem in arrayToBe:
    if elem in A:
        happy+=1
    
    if elem in B:
        happy-=1

print(happy)

# Symmetric Difference

N = int(input())
setA = set(map(int, input().split()))

M = int(input())
setB = set(map(int, input().split()))

sym = sorted(set(setA.union(setB) - setA.intersection(setB)))
for num in sym:
    print(num)

# Set .add()

N = int(input())

countries = set()
for country in range(N):
    countries.add(input())

print(len(countries))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))

ops = int(input())

for _ in range(ops):
    cmd = input().split()
    
    if cmd[0] == "pop":
        s.pop()
    elif cmd[0] == "remove":
        s.remove(int(cmd[1]))
    else:
        s.discard(int(cmd[1]))

print(sum(s))

# Set .union() Operation

students_en = int(input())
arrayEN = set(map(int, input().split()))

students_fr = int(input())
arrayFR = set(map(int, input().split()))

print(len(arrayEN.union(arrayFR)))

# Set .intersection() Operation

students_en = int(input())
arrayEN = set(map(int, input().split()))

students_fr = int(input())
arrayFR = set(map(int, input().split()))

print(len(arrayEN.intersection(arrayFR)))

# Set .difference() Operation

students_en = int(input())
arrayEN = set(map(int, input().split()))

students_fr = int(input())
arrayFR = set(map(int, input().split()))

print(len(arrayEN.difference(arrayFR)))

# Set .symmetric_difference() Operation

students_en = int(input())
arrayEN = set(map(int, input().split()))

students_fr = int(input())
arrayFR = set(map(int, input().split()))

print(len(arrayEN.symmetric_difference(arrayFR)))

# Set Mutations

A = int(input())
setA = set(map(int, input().split()))

N = int(input())

for op in range(N):
    cmd = input().split()
    setB = set(map(int, input().split()))

    if cmd[0] == "intersection_update":
        setA.intersection_update(setB)
    elif cmd[0] == "update":
        setA.update(setB)
    elif cmd[0] == "symmetric_difference_update":
        setA.symmetric_difference_update(setB)
    else:
        setA.difference_update(setB)

print(sum(setA))

# The Captain's Room

S = int(input())
lst_room = map(int, input().split())

captain_room = set()
family_room = set()

for r in lst_room:
    if r in captain_room:
        family_room.add(r)
    else:
        captain_room.add(r)

print(list(captain_room.difference(family_room))[0])

# Check Subset

test_cases = int(input())

for _ in range(test_cases):
    sizeA = int(input())
    setA = set(map(int, input().split()))
    
    sizeB = int(input())
    setB = set(map(int, input().split()))

    print(setB.intersection(setA) == setA)

# Check Strict Superset

setStrictA = set(map(int, input().split()))

N = int(input())
flg = 0
for _ in range(N):
    setStrictX = set(map(int, input().split()))

    if setStrictA.intersection(setStrictX) == setStrictX:
        flg += 1
    
print(("True" if flg == N else "False")) 

# collections.Counter()

from collections import Counter

quantity_shoes = int(input())
list_size_shoes = Counter(list(map(int, input().split())))

customers = int(input())
money_earned = 0
for c in range(customers):
    size, price = map(int, input().split())
    if list_size_shoes[size]:
        money_earned += price
        list_size_shoes[size] -= 1

print(money_earned)

# DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)

sizeA, sizeB = map(int, input().split())
for elemA in range(1, sizeA+1):
    d[input()].append(str(elemA))

for elemB in range(sizeB):
    print(' '.join(d[input()]) or -1)

# Collections.namedtuple()

from collections import namedtuple

num_students = int(input())
students = namedtuple('student', input().split())
counter = 0
for _ in range(num_students):
    c1, c2, c3, c4 = input().split()
    s = students(c1, c2, c3, c4)
    counter += int(s.MARKS)

print("{0:.2f}".format(counter/num_students))

# Collections.OrderedDict()

from collections import OrderedDict

ordered_dictionary = OrderedDict()
for _ in range(int(input())):
    *item, price = input().split()
    if len(item) > 0:
        str1 = " "
        item = str1.join(item)
        if ordered_dictionary.get(item):
            ordered_dictionary[item] += int(price)
        else:
            ordered_dictionary[item] = int(price)
    else:
        if ordered_dictionary.get(item[0]):
            ordered_dictionary[item[0]] += int(price)
        else:
            ordered_dictionary[item[0]] = int(price)

for item in ordered_dictionary:
    print(item, ordered_dictionary[item])

# Word Order

from collections import defaultdict
d = defaultdict(list)

for _ in range(int(input())):
    elem = input()
    d[elem].append(1)

print(len(d.items()))
    
for i in d.items():
    print(len(i[1]), end= " ")

# Collections.deque()

from collections import deque
d = deque()

for _ in range(int(input())):
    cmd = input().split()
    if cmd[0] == "append":
        d.append(cmd[1])

    if cmd[0] == "pop":
        d.pop()

    if cmd[0] == "appendleft":
        d.appendleft(cmd[1])
    
    if cmd[0] == "popleft":
        d.popleft()

for num in d:
    print(num, end=" ")

# Company Logo

if __name__ == '__main__':
    s = sorted(input())
    ordered_dictionary = {}

    for c in range(len(s)):
        if ordered_dictionary.get(s[c]):
            ordered_dictionary[s[c]] += 1
        else:
            ordered_dictionary[s[c]] = 1

    max_value = -1
    sy = "o"
    for _ in range(3):
        for item in ordered_dictionary:
            if ordered_dictionary[item] > max_value and ordered_dictionary[item] != -999:
                max_value = ordered_dictionary[item]
                sy = item
        print(sy, max_value)
        ordered_dictionary[sy] = -999
        max_value = -1

# Piling Up!

#(I saw the code in terms logic way, then I created the function)
n = int(input())
for _ in range(n): 
    _, cubes = input(), list(map(int, input().split()))
    min_index = cubes.index(min(cubes))
    print("Yes" if sorted(cubes[:min_index], reverse=True) == cubes[:min_index] and sorted(cubes[min_index:]) == cubes[min_index:] else "No")

# Calendar Module

import calendar

month, day, year = map(int, input().split())
print((calendar.day_name[calendar.weekday(year, month, day)].upper()))

# Time Delta

from datetime import datetime

def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds()))) 

# Exceptions

for _ in range(int(input())):
    try:
        a,b = map(int, input().split()) 
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)

# Zipped!

students, subjects = map(int, input().split())
subj1 = []
for _ in range(subjects):
    subj1.append(map(float, input().split()))

zipped = zip(*subj1)
for i in zipped:
    print(sum(i)/len(i))

# Athlete Sort

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])

    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    max_value = arr[0][k]
    for i in range(1, m):
        if arr[m][k] > max_value:
            max_value = arr[i][k]

# ginortS

s = input()
sorted_s_lower = []
sorted_s_upper = []
sorted_s_digit = []
odds = []
even = []

for character in s:
    if character.islower():
        sorted_s_lower.append(character)
    elif character.isupper():
        sorted_s_upper.append(character)
    elif character.isdigit():
        if int(character) % 2 == 0:
            odds.append(character)
        else:
            even.append(character)
        sorted_s_digit = sorted(even) + sorted(odds)

print(''.join((sorted(sorted_s_lower) + sorted(sorted_s_upper) + sorted_s_digit)))

# Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    fib_list = []
    for i in range(n):
        if len(fib_list) == 0:
            fib_list.append(0)
        elif len(fib_list) == 1:
            fib_list.append(1)
        else:
            fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Detect Floating Point Number

import re

for _ in range(int(input())):
    print(bool(re.match(r'^[+-]?[0-9]*\.[0-9]+$', input())))

# Re.split()

regex_pattern = r"[,.]+"

import re
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() & Groupdict()

import re

s = (re.search(r'([a-z0-9])\1+', input()))
if s:
    print(s.group(1))
else:
    print("-1")

# Re.findall() & Re.finditer()

import re

lst = re.findall(r"(?<=[qwrtypsdfghjklzxcvbnm])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnm])", input())

if len(lst) >= 1:
    for i in lst:
        print(i)
else:
    print("-1")

# Re.start() & Re.end()

#(I saw the code, I am grateful for the solutions in the network because I understand well the other functions of re library! Py documentation)
import re 

string = input()
substring = input()

m = re.search(substring, string)
path = re.compile(substring)
if m:
    while m:
        print("({0}, {1})".format(m.start(),m.end()-1))
        m = path.search(string, m.start()+1)
else:
    print("(-1, -1)")

# Regex Substitution

import re

for _ in range(int(input())):
      orType = re.compile(r'(?<= )(\|\|)(?= )')
      andType = re.compile(r'(?<= )(&&)(?= )')
      
      print(orType.sub('or', andType.sub('and', input())))

# Validating Roman Numerals

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating phone numbers

import re

for _ in range(int(input())):
    if re.match(r'[789]\d{9}$', input()):   
        print('YES') 
    else:  
        print('NO') 

# Validating and Parsing Email Addresses

import re

#(I saw the code, I am grateful for the solutions in the network because I understand well how to make a complex regex like this! (There are regex pattern strange other easy to understand))
for _ in range(int(input())):
    x, y = input().split(' ') 
    if re.match(r"<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>", y):
        print(x,y)

# Hex Color Code

import re 

for _ in range(int(input())): 
    rgx = re.findall(r".(#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3})", input()) 
    if rgx:
        for i in rgx:
            print(i)

# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs: 
            print("-> " + attr[0] + " >", attr[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print("-> " + attr[0] + " >", attr[1])

parser = MyHTMLParser()

for _ in range(int(input())):
    parser.feed(input().strip())

# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" in data:
                print('>>> Multi-line Comment')
                print(data)
        else:
                print('>>> Single-line Comment')
                print(data)
  
    def handle_data(self, data):
        if '\n' not in data:
            print(">>> Data")
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("-> " + attr[0], ">", attr[1])

parser = MyHTMLParser()

for i in range(int(input())):
    parser.feed(input())

# Validating UID

import re

#(The same thing about my last comment)
no_repeats = r"(?!.*(.).*\1)"
two_or_more_upper = r"(.*[A-Z]){2,}"
three_or_more_digits = r"(.*\d){3,}"
ten_alphanumerics = r"[a-zA-Z0-9]{10}"
filters = no_repeats, two_or_more_upper, three_or_more_digits, ten_alphanumerics

for UID in [input() for _ in range(int(input()))]:
    print("Valid" if(all(re.match(fs, UID) for fs in filters)) else "Invalid" )

# Validating Credit Card Numbers

import re

valid_structure = r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$"
for _ in range(int(input())):
    string = input()
    if re.match(valid_structure, string) and not re.search(r"([\d])\1\1\1", string.replace("-", "")):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"	# Do not delete 'r'.

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Matrix Script

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])
m = int(first_multiple_input[1])

a, b = [], ""

for _ in range(n): #throw away matrix case
    a.append(input())

for z in zip(*a): #zip "a" to obtain the columns like rows!
    b += "".join(z)

print(re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", b))

# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    sm = 0
    for children in node.iter():
        sm += len(children.attrib)
    return sm

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1) 
        
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
       f(["+91 " + digits[-10:-5] + " " + digits[-5:] for digits in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 


# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        return map(f, people.sort(key=operator.itemgetter(2)))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# Arrays

import numpy

def arrays(arr):
    a = numpy.array(arr, float)
    return a[::-1]

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape

import numpy

num_lst = numpy.array(input().split(), int)
print(numpy.reshape(num_lst, (3,3)))

# Transpose and Flatten

import numpy

rows, columns = map(int, input().split())

lst_npy = numpy.array([input().split() for _ in range(rows)], int)
print(lst_npy.transpose())
print(lst_npy.flatten())

# Concatenate

import numpy

n, m, p = map(int, input().split())
lst_num = numpy.array([input().split() for _ in range(n)], int)
lst2_num = numpy.array([input().split() for _ in range(m)], int)

print(numpy.concatenate((lst_num, lst2_num), axis=0))

# Zeros and Ones

import numpy

t = tuple(map(int, input().split()))

print( numpy.zeros((t), dtype = numpy.int) )
print( numpy.ones((t), dtype = numpy.int) )

# Eye and Identity

import numpy

n, m = map(int, input().split())
print(str(numpy.eye(n, m, k=0)).replace('1', ' 1').replace('0', ' 0'))

# Array Mathematics

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), dtype = int)
for i in range(n):
    a[i] = numpy.array([input().split()], int)

b = numpy.zeros((n, m), dtype = int)
for i in range(n):
    b[i] = numpy.array([input().split()], int)

print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

# Floor, Ceil and Rint

import numpy

numpy.set_printoptions(sign=' ')

my_array = numpy.array([input().split()], dtype=float)
print(numpy.floor(*my_array))
print(numpy.ceil(*my_array))
print(numpy.rint(*my_array))

# Sum and Prod

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.prod(numpy.sum(a, axis = 0), axis = None))

# Min and Max

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.max(numpy.min(a, axis = 1), axis = None))

# Mean, Var, and Std

import numpy

numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(numpy.std(a, axis=None))

# Dot and Cross

import numpy

n = int(input())

a = numpy.zeros((n, n), int)
for i in range(n):
    a[i] = input().split()

b = numpy.zeros((n, n), int)
for i in range(n):
    b[i] = input().split()

print(numpy.dot(a, b))

# Inner and Outer

import numpy

a = numpy.array(input().split(), dtype=int)
b = numpy.array(input().split(), dtype=int)

print(numpy.inner(a,b))
print(numpy.outer(a,b))

# Polynomials

import numpy

lst = list(map(float, input().split()))
print(numpy.polyval(lst, int(input())))

# Linear Algebra

import numpy

numpy.set_printoptions(legacy='1.13')

a = numpy.array([input().split() for _ in range(int(input()))], dtype=float)

print(numpy.linalg.det(a))

# Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    max_candle = max(candles)

    counter = 0
    for c in candles:
        if c == max_candle:
            counter+=1

    return counter

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    return("YES" if (v1 > v2) and (x2-x1)%(v1-v2) == 0 else "NO")

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    start_people = 5
    cumulative = 0
    for _ in range(n):
        start_people = math.floor(start_people/2)
        cumulative += start_people
        start_people = start_people*3
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    if(len(list(map(int, n))) > 1):
        new_digit = sum(list(map(int, n))) * k
        return superDigit(str(new_digit), 1)
    else:
        return int(n)
  
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def stamp(arr):
    for elem in arr:
        print(elem, end=" ")
    print("")

def insertionSort1(n, arr):
    for i in range(1, len(arr)):
        x = arr[i]
        j = i-1
        while j >=0 and x < arr[j]:
            arr[j+1] = arr[j]
            stamp(arr)
            j -= 1
        arr[j+1] = x
    stamp(arr)
  
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def stamp(arr):
    for elem in arr:
        print(elem, end=" ")
    print("")

def insertionSort2(n, arr):
    for i in range(1, len(arr)):
        x = arr[i]
        j = i-1
        while j >= 0 and x < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = x
        stamp(arr)
        
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

