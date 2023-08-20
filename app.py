import numpy as np
from flask import Flask, request, jsonify, render_template
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import logging
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/summarize', methods=['POST','GET'])
def summarize():
    if request.method == 'POST':
        # Get user input for the query and the number of search results
        query = request.form.get('query')
        num_results = request.form.get('doccount')
        x=request.form.get('wordcount')
          # Use the search function to retrieve the URLs of the search results
        # search_results = search(query, num_results=num_results)

        # # Loop through the search results and print the URLs
        # for url in search_results:
        #     print(url)
        link=['https://www.cnet.com/tech/mobile/best-iphone/','https://www.techradar.com/news/best-iphone','https://www.tomsguide.com/us/best-apple-iphone,review-6348.html']
            # link.append(url)
        # Define the URLs to scrape
        urls = link
        # Create an empty list to store the text content of the <p> tags for each URL
        p_contents = []
        # Loop through each URL and extract the text content of the <p> tags
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            p_contents.append([p.get_text() for p in paragraphs])
        # Print the text content of the <p> tags for each URL
        for i, url in enumerate(urls):
            print(f"Contents of <p> tags for URL {i + 1} ({url}):")
            for p_content in p_contents[i]:
                print(p_content)
                print()
        # Initialize stop words
        stop_words = set(stopwords.words('english'))

        def preprocess(text):
            # Join list of strings into a single string
            cleaned_text = " ".join(text)
            # Remove unwanted characters
            cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', '')
            # Tokenize the text
            tokens = word_tokenize(cleaned_text)
            # Remove stop words
            preprocessed_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            # Join tokens back into a string
            preprocessed_text = " ".join(preprocessed_tokens)
            return preprocessed_text

        # Preprocess each paragraph in p_contents
        preprocessed_p_contents = [preprocess(p) for p in p_contents]
        # Join all strings in the list into a single paragraph
        summary = ' '.join(preprocessed_p_contents)

        # Generate a word cloud object
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate(summary)

        # Plot the word cloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        text = preprocessed_p_contents
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 7)  # 3 is the number of sentences in the summary
        summary_list = [str(sentence)[10:-2] for sentence in summary]
        # Join all strings in the list into a single paragraph
        paragraph = ' '.join(summary_list)

        # Print the resulting paragraph
        print(paragraph)
        # define a list of sentences
        sentences = summary_list

        # create a CountVectorizer object
        vectorizer = CountVectorizer(stop_words='english')

        # fit the vectorizer on the sentences
        vectorizer.fit(sentences)

        # transform the sentences into a matrix of word counts
        matrix = vectorizer.transform(sentences)

        # create a heatmap of the matrix
        sns.heatmap(matrix.toarray(), annot=True, xticklabels=vectorizer.get_feature_names(),
                    yticklabels=sentences,cmap='Blues')

        # show the plot
        plt.figure(figsize=(12, 8))
        sns.set(font_scale=1.2)
        plt.show()


        # Set logging level to WARNING to suppress all warnings
        logging.getLogger("transformers").setLevel(logging.WARNING)

        # Define summarizer pipeline
        # hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
        # summarizer = pipeline("summarization", model=hub_model_id)
        summarizer = pipeline("summarization", model="t5-small")

        # Define article to summarize
        ARTICLE = paragraph
        # Call summarizer pipeline on article min_length=x
        summary = summarizer(ARTICLE,do_sample=False)

        # Print the summary
        # print(summary)
        # # Return the summary as a response to the POST request
        # return jsonify(summary=summary)

    return render_template('index2.html', result=ARTICLE)
if __name__ == "__main__":
    app.run(debug=True)
