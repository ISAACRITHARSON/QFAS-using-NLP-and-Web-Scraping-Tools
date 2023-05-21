# Abstractive-Query-Focused-Multi-Document-Summarizer
The system uses a variety of tools and methods to automate the summary process and increase users' access to information, including web scraping, natural language processing, and transformer models. A user's inquiry, the amount of words to be condensed, and the number of documents to be cited can all be entered into the system. The most pertinent webpages are then retrieved via a Google search engine API integration, and tag web scraping is carried out using the beautiful soup (bs4) and selenium frameworks.The pre-processed data is tokenized using Auto tokenizer, stop words are removed from the scraped data, and matplotlib and seaborn are used to visualise frequency matrices and word clouds. The pipeline summarizer in the system uses a transformer model called "mt5-small Pretrained." By ranking the terms according to their frequency, the transformer model creates a summary of the text that is logical, succinct, and pertinent to the user's question. The result of the system is a well-organized summary that gathers the crucial details from several sources. 


 
## Key-Terms
- Abstractive summarization
- multi-document summarization 
- Query-focused summarization
- Transformer models
- Web scraping
- API.




## Run Locally

Clone the project

```bash
  git clone https://github.com/staroIR11/Query-Focused-Multi-Document-Summarizer
```





## Tech Stack

**Hardware:** Dell G15 Laptop, Graphics: Rtx 3050, Processor: Amd Ryzen 5  

**Software:** Visual Studio Code, Jupyter Notebooks, NLTK, Flask, BS4, Selenium Libraries 

## Support

For support, email isaacritharson@karunya.edu.in 

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)]()
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/isaac-ritharson-p-36924b209/)



## References

 -[1]	Ana B. Rios-Alvarado at al.,2929 Mining information from sentences through Semantic Web data and Information Extraction tasks, Journal of Information Science2022, Vol. 48(1) 3–20, DOI:10.1177/0165551520934387]
 
 -[2]	Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://aclanthology.org/K16-1028) (Nallapati et al., CoNLL 2016)

-[3]	N. Andhale and L. A. Bewoor, "An overview of Text Summarization techniques," 2016 International Conference on Computing Communication Control and automation (ICCUBEA), Pune, India, 2016, pp. 1-7, doi: 10.1109/ICCUBEA.2016.7860024.

-[4]	Tal Baumel and Matan Eyal and Michael Elhadad, Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models, 2018

-[5]	J. Du and Y. Gao, "Query-focused Abstractive Summarization via Question-answering Model," 2021 IEEE International Conference on Big Knowledge (ICBK), Auckland, New Zealand, 2021, pp. 440-447, doi: 10.1109/ICKG52313.2021.00065.


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


-Authors: Isaac Ritharson P*, J.Anitha, Sujitha Juliet
