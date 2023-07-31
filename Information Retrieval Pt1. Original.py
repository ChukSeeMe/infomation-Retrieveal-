#!/usr/bin/env python
# coding: utf-8

# # Importing Liberies
# 
# Started by imported the liberaries that i will need for the task

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import datetime
import string
import feedparser
import tkinter as tk
import schedule
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import json
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[2]:



def crawl_publications():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"
    publications = []

    for page in range(1, 7):
        url = f"{base_url}?page={page}"

        response = requests.get(url)

        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract publication information
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None

                publication_link_element = element.select_one('.title > a')
                publication_link = publication_link_element['href'] if publication_link_element else None

                publication_year_element = element.select_one('.date')
                publication_year = publication_year_element.text.strip() if publication_year_element else None

                author_name_element = element.select_one('.person')
                author_name = author_name_element.text.strip() if author_name_element else None

                author_profile_link_element = element.select_one('.person a')
                author_profile_link = author_profile_link_element['href'] if author_profile_link_element else None

                # Store the publication data in a dictionary
                publication_data = {
                    'Title': publication_title,
                    'Publication Link': publication_link,
                    'Publication Year': publication_year,
                    'Author': author_name,
                    'Author Profile Link': author_profile_link
                }

                # Append the publication data to the list
                publications.append(publication_data)

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code)

        # Be polite and wait for a short duration between requests
        time.sleep(1)

    return publications

# Call the web scraping function and get the publications data
publications_data = crawl_publications()

# Display all the publication data
for publication in publications_data:
    print("Title:", publication['Title'])
    print("Publication Year:", publication['Publication Year'])
    print("Author:", publication['Author'])
    print("Publication Link:", publication['Publication Link'])
    print("Author Profile Link:", publication['Author Profile Link'])
    print("\n")


# # Number Of Staff Publications Crawled

# In[3]:




def crawl_publications():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"
    publications = []
    authors = set()  # Use a set to store unique author names

    for page in range(1, 7):
        url = f"{base_url}?page={page}"

        response = requests.get(url)

        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract publication information
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None

                publication_link_element = element.select_one('.title > a')
                publication_link = publication_link_element['href'] if publication_link_element else None

                publication_year_element = element.select_one('.date')
                publication_year = publication_year_element.text.strip() if publication_year_element else None

                author_name_element = element.select_one('.person')
                author_name = author_name_element.text.strip() if author_name_element else None

                author_profile_link_element = element.select_one('.person a')
                author_profile_link = author_profile_link_element['href'] if author_profile_link_element else None

                # Store the publication data in a dictionary
                publication_data = {
                    'Title': publication_title,
                    'Publication Link': publication_link,
                    'Publication Year': publication_year,
                    'Author': author_name,
                    'Author Profile Link': author_profile_link
                }

                # Append the publication data to the list
                publications.append(publication_data)

                # Add the author name to the set of authors
                if author_name:
                    authors.add(author_name)

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code)

        # Be polite and wait for a short duration between requests
        time.sleep(1)

    return publications, authors

# Call the web scraping function and get the publications data and authors
publications_data, authors_set = crawl_publications()

# Count the number of unique authors whose publications are crawled
num_unique_authors = len(authors_set)

print("Number Of Staff Publications Crawled:", num_unique_authors)


# In[4]:



def crawl_and_print_publications():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"

    for page in range(1, 7):
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        if response.status_code == 200:     # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')      # Extracting publication informations:
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None
                publication_link_element = element.select_one('.title > a')
                publication_link = publication_link_element['href'] if publication_link_element else None
                publication_year_element = element.select_one('.date')
                publication_year = publication_year_element.text.strip() if publication_year_element else None
                author_name_element = element.select_one('.person')
                author_name = author_name_element.text.strip() if author_name_element else None
                author_profile_link_element = element.select_one('.person')
                author_profile_link = author_profile_link_element['href'] if author_profile_link_element else None

                # Extract keywords available
                keywords_element = element.select_one('.keywords')
                keywords = keywords_element.text.strip() if keywords_element else 'Not available'
                print(f"Title: {publication_title}")    # Print out the publication details
                print(f"Publication Link: {publication_link}")
                print(f"Publication Year: {publication_year}")
                print(f"Author: {author_name}")
                print(f"Author Profile Link: {author_profile_link}")
                print(f"Keywords: {keywords}")
                print("\n")

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code) 
            time.sleep(1)         # Call the function to crawl 
                                  # waiting for a short duration between requests politely
crawl_and_print_publications()    # print out the publications:


# # Indexing

# In[5]:



def crawl_and_build_inverted_index():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"
    inverted_index = {}

    for page in range(1, 7):
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None

                keywords_element = element.select_one('.keywords')
                keywords = keywords_element.text.strip() if keywords_element else 'Not available'

                # Process keywords and update the inverted index
                keyword_list = keywords.split(', ')
                for keyword in keyword_list:
                    keyword = keyword.lower()  # Convert keyword to lowercase
                    if keyword not in inverted_index:
                        inverted_index[keyword] = []  # Initialize an empty list for the keyword
                    inverted_index[keyword].append(publication_title)  # Add publication to the list

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code)

        # Be polite and wait for a short duration between requests
        time.sleep(1)

    return inverted_index

inverted_index = crawl_and_build_inverted_index()

# Print the inverted index
for keyword, publications in inverted_index.items():
    print(f"Keyword: {keyword}")
    print("Publications:")
    for publication in publications:
        print(f"  - {publication}")
    print("\n")


# In[9]:


import requests
from bs4 import BeautifulSoup
import time

def crawl_and_build_inverted_index():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"
    inverted_index = {}

    for page in range(1, 7):
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None

                link_element = element.select_one('.title > a')
                link = link_element['href'] if link_element else None

                keywords_element = element.select_one('.keywords')
                keywords = keywords_element.text.strip() if keywords_element else 'Not available'

                # Process keywords and update the inverted index
                keyword_list = keywords.split(', ')
                for keyword in keyword_list:
                    keyword = keyword.lower()  # Convert keyword to lowercase
                    if keyword not in inverted_index:
                        inverted_index[keyword] = []  # Initialize an empty list for the keyword
                    inverted_index[keyword].append({'title': publication_title, 'link': link})  # Add publication details to the list

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code)

        # Be polite and wait for a short duration between requests
        time.sleep(1)

    return inverted_index

inverted_index = crawl_and_build_inverted_index()

# Print the inverted index
for keyword, publications in inverted_index.items():
    print(f"Keyword: {keyword}")
    print("Publications:")
    for publication in publications:
        print(f"  - {publication['title']}")
        print(f"    Link: {publication['link']}")
    print("\n")


# # Creating Search Interface and Querrying 

# In[ ]:


import requests
from bs4 import BeautifulSoup
import time
import sqlite3
import re
import tkinter as tk
import webbrowser


def crawl_publications():
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning"

    publications = []

    for page in range(1, 7):
        url = f"{base_url}?page={page}"

        response = requests.get(url)

        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract publication information
            publication_elements = soup.select(".rendering")

            for element in publication_elements:
                publication_title_element = element.select_one('.title > a')
                publication_title = publication_title_element.text.strip() if publication_title_element else None

                publication_link_element = element.select_one('.title > a')
                publication_link = publication_link_element['href'] if publication_link_element else None

                publication_year_element = element.select_one('.date')
                publication_year = publication_year_element.text.strip() if publication_year_element else None

                author_name_element = element.select_one('.person')
                author_name = author_name_element.text.strip() if author_name_element else None

                author_profile_link_element = element.select_one('.person')
                author_profile_link = author_profile_link_element['href'] if author_profile_link_element else None

                # Store the publication data in a dictionary
                publication_data = {
                    'Title': publication_title,
                    'Publication Link': publication_link,
                    'Publication Year': publication_year,
                    'Author': author_name,
                    'Author Profile Link': author_profile_link
                }

                # Append the publication data to the list
                publications.append(publication_data)

        else:
            print(f"Failed to load page {page}. Status code:", response.status_code)

        # Be polite and wait for a short duration between requests
        time.sleep(1)

    return publications


def search_publications(query, publications):
    results = []

    # Pre-process the query
    query = preprocess_query(query)

    for publication in publications:
        title = publication['Title']

        # Pre-process the publication title
        title = preprocess_text(title)

        if title and query in title:
            results.append(publication)

    return results


def preprocess_query(query):
    # Convert the query to lowercase and remove special characters
    query = re.sub(r'[^\w\s]', '', query.lower())
    return query


def preprocess_text(text):
    if text:
        # Convert the text to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text
    else:
        return ''


# Database functions for data indexing
def create_table():
    conn = sqlite3.connect('publications.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY,
            title TEXT,
            link TEXT,
            year TEXT,
            author TEXT,
            profile_link TEXT
        )
    ''')

    conn.commit()
    conn.close()


def insert_data(publications):
    conn = sqlite3.connect('publications.db')
    cursor = conn.cursor()

    for publication in publications:
        title = publication['Title']
        link = publication['Publication Link']
        year = publication['Publication Year']
        author = publication['Author']
        profile_link = publication['Author Profile Link']

        cursor.execute('''
            INSERT INTO publications (title, link, year, author, profile_link)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, link, year, author, profile_link))

    conn.commit()
    conn.close()


def fetch_data():
    conn = sqlite3.connect('publications.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM publications')
    rows = cursor.fetchall()

    data = []
    for row in rows:
        publication_data = {
            'Title': row[1],
            'Publication Link': row[2],
            'Publication Year': row[3],
            'Author': row[4],
            'Author Profile Link': row[5]
        }
        data.append(publication_data)

    conn.close()
    return data


def display_search_results(results):
    result_text.delete("1.0", tk.END)  # Clear previous results

    if len(results) > 0:
        for publication in results:
            result_text.insert(tk.END, "Title: " + publication['Title'] + "\n")
            result_text.insert(tk.END, "Publication Year: " + publication['Publication Year'] + "\n")

            if publication['Author']:
                result_text.insert(tk.END, "Author: " + publication['Author'] + "\n")
            else:
                result_text.insert(tk.END, "Author: Not available\n")

            result_text.insert(tk.END, "Publication Link: ")
            result_text.insert(tk.END, publication['Publication Link'] + "\n", "publication_link")

            if publication['Author Profile Link']:
                result_text.insert(tk.END, "Author Profile Link: ")
                result_text.insert(tk.END, publication['Author Profile Link'] + "\n", "profile_link")
            else:
                result_text.insert(tk.END, "Author Profile Link: Not available\n")

            result_text.insert(tk.END, "\n")

        # Add tag configurations for hyperlink-like styling
        result_text.tag_configure("publication_link", foreground="blue", underline=True)
        result_text.tag_configure("profile_link", foreground="blue", underline=True)

        # Bind the click events to the hyperlinks
        result_text.tag_bind("publication_link", "<Button-1>", lambda event: open_link(event, results, 'Publication Link'))
        result_text.tag_bind("profile_link", "<Button-1>", lambda event: open_link(event, results, 'Author Profile Link'))

        # Make the links clickable
        result_text.config(state=tk.DISABLED)
        result_text.config(cursor="hand2")

    else:
        result_text.insert(tk.END, "No results found for the search query.")


def open_link(event, results, link_key):
    index = result_text.index(tk.CURRENT)
    line_number = int(index.split('.')[0]) - 1
    clicked_link = results[line_number][link_key]

    if clicked_link:
        webbrowser.open_new_tab(clicked_link)


def search_interface():
    # Web crawling (run once per week and update index)
    publications = crawl_publications()
    create_table()
    insert_data(publications)

    window = tk.Tk()
    window.title("Publication Search")
    window.geometry("700x700")

    def search():
        query = entry.get()
        data = fetch_data()
        search_results = search_publications(query, data)
        display_search_results(search_results)

    label = tk.Label(window, text="Enter your search query:")
    label.pack()

    entry = tk.Entry(window)
    entry.pack()

    button = tk.Button(window, text="Search", command=search)
    button.pack()

    global result_text  # Define result_text as a global variable
    result_text = tk.Text(window)
    result_text.pack()

    window.mainloop()


# Run the search interface
search_interface()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




