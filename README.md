# Search Engine Project README

## Overview

This project implements a custom search engine capable of querying a dataset and returning relevant results. It comprises two main components:

1. **Search Logic (`my_search.py`):** Implements the search algorithm, including preprocessing queries, calculating TF-IDF weights, and ranking documents based on cosine similarity and BM25 score.
   
2. **Web API (`search_frontend.py`):** A Flask application that provides a RESTful API for interacting with the search engine. It supports different search modes, including searching by body content, title, and anchor text, and can return PageRank values and page view statistics.

## Setup

### Requirements

- Python 3.x
- Flask
- NLTK
- Additional Python libraries as needed (requests, math, collections, re)

### Installation

1. Clone the repository to your local machine or server.
2. Install the required Python packages:

```bash
pip install flask nltk requests
```

3. Download the necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
```

4. Ensure you have access to the dataset and the GCP (Google Cloud Platform) bucket mentioned in `my_search.py`.

## Running the Application

To start the Flask application, navigate to the directory containing `search_frontend.py` and run:

```bash
python search_frontend.py
```

The server will start on port 8080 and can be accessed through `http://localhost:8080` or the appropriate domain/IP address if deployed on a server.

## API Usage

The web API supports several endpoints for different search functionalities:

- **General Search**: `/search?query=<your_query>`

Replace `<your_query>` with the search terms of your interest.

## Contributing

We welcome contributions to improve the search engine or extend its functionalities. Please feel free to submit issues or pull requests.

