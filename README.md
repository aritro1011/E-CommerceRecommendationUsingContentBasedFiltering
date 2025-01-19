# E-Commerce Recommendation System

This project is a content-based filtering recommendation system for an e-commerce platform. It suggests products to users based on their preferences by analyzing product descriptions and features.

## Features
- **Content-Based Filtering**: Recommendations are made using similarities between products.
- **TF-IDF Vectorization**: Utilizes TF-IDF to analyze and compare product descriptions.

---

## How It Works
1. **Data Preprocessing**: 
   - Cleans and preprocesses the data to handle missing values, remove duplicates, and normalize features.
   - Text data is vectorized using TF-IDF for similarity calculation.

2. **Recommendation Logic**: 
   - Calculates the cosine similarity between products.
   - Suggests the most similar products based on the user's input.



---

## Installation

### Prerequisites
- Python 3.x
- Flask
- Postman (or any API testing tool)

### Clone the Repository
```bash
git clone <https://github.com/aritro1011/E-CommerceRecommendationUsingContentBasedFiltering>
cd <E-CommerceRecommendationUsingContentBasedFiltering>
```
## Install Required Libraries

Install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
Running the Application
Start the Flask App:
Run the app locally with:

```bash

python main.py
```
By default, the app will run on http://127.0.0.1:5000.  

File Structure  

*main.py                # Flask application  

*requirements.txt      # Dependencies  

*amazon.csv            #dataset  

*program.py            #python code    

*templates/            # HTML templates   

*README.md             # Project documentation
