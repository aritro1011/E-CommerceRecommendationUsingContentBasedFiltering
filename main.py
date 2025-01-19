from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import nltk

app = Flask(__name__)

# Ensure nltk resources are downloaded
nltk.download('stopwords')

def load_and_clean_data():
    # Step 1: Load the dataset
    df = pd.read_csv('amazon.csv')

    # Step 2: Clean the 'actual_price' and 'discounted_price' columns
    df['actual_price'] = df['actual_price'].replace({'₹': '', ',': '', '%': ''}, regex=True).astype(float)
    df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': '', '%': ''}, regex=True).astype(float)

    # Step 3: Fill missing values for each numeric column separately
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].mean())  # Fill missing values with column mean

    # Step 4: Scale numerical features
    numerical_features = ['actual_price', 'discounted_price']  # Add more numerical features if needed
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Step 5: Clean and process textual data
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()  # Remove non-alphabetic characters
        text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
        return text

    # Apply text cleaning for relevant columns
    df['review_content'] = df['review_content'].apply(clean_text)
    df['about_product'] = df['about_product'].apply(clean_text)

    # Step 6: Vectorization for textual data (review_content, about_product, category)
    vectorizer_review_content = TfidfVectorizer(max_features=500)
    review_content_tfidf = vectorizer_review_content.fit_transform(df['review_content'])
    review_content_tfidf_df = pd.DataFrame(review_content_tfidf.toarray(), columns=vectorizer_review_content.get_feature_names_out())
    df = pd.concat([df, review_content_tfidf_df], axis=1)

    vectorizer_about_product = TfidfVectorizer(max_features=500)
    about_product_tfidf = vectorizer_about_product.fit_transform(df['about_product'])
    about_product_tfidf_df = pd.DataFrame(about_product_tfidf.toarray(), columns=vectorizer_about_product.get_feature_names_out())
    df = pd.concat([df, about_product_tfidf_df], axis=1)

    vectorizer_category = TfidfVectorizer(max_features=500)
    category_tfidf = vectorizer_category.fit_transform(df['category'])
    category_tfidf_df = pd.DataFrame(category_tfidf.toarray(), columns=vectorizer_category.get_feature_names_out())
    df = pd.concat([df, category_tfidf_df], axis=1)

    # Step 7: Keep only relevant product info and drop other columns
    product_info = df[['product_id', 'product_name', 'img_link', 'product_link']]
    df = df.drop(columns=['product_id', 'product_name', 'user_id', 'user_name', 'review_id', 'review_title', 'category', 'about_product', 'review_content', 'img_link', 'product_link'])

    # Fill missing values only in numeric columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Step 8: Merge cleaned data with product information
    df_with_name = pd.concat([product_info[['product_id', 'product_name', 'img_link', 'product_link']], df], axis=1)

    # Step 9: Compute cosine similarity based on selected features
    cosine_sim = cosine_similarity(df.select_dtypes(include=['float64', 'int64']))

    return df_with_name, cosine_sim

# Load the data and model
df_with_name, cosine_sim = load_and_clean_data()

# Define a function for recommendation
def recommend_products(product_id, top_n=5):
    if product_id not in df_with_name['product_id'].values:
        return "Product ID not found in the dataset."

    product_idx = df_with_name[df_with_name['product_id'] == product_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[product_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_products = [
        (df_with_name['product_id'][i[0]], df_with_name['product_name'][i[0]], df_with_name['img_link'][i[0]], df_with_name['product_link'][i[0]])
        for i in sorted_scores 
        if df_with_name['product_id'][i[0]] != product_id and 
           pd.notna(df_with_name['product_id'][i[0]]) and 
           pd.notna(df_with_name['product_name'][i[0]]) and
           pd.notna(df_with_name['img_link'][i[0]]) and
           pd.notna(df_with_name['product_link'][i[0]])
    ]

    if not top_products:
        return "No valid recommended products found."
    return top_products[:top_n]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    product_id_input = request.form['product_id']
    recommended_products = recommend_products(product_id=product_id_input, top_n=5)

    if isinstance(recommended_products, str):
        return jsonify({'error': recommended_products})

    # Fetching the entered product information
    entered_product_info = df_with_name[df_with_name['product_id'] == product_id_input].iloc[0]
    entered_product_name = entered_product_info['product_name']
    entered_product_img_link = entered_product_info['img_link']
    entered_product_link = entered_product_info['product_link']

    # Prepare recommendations for rendering in HTML
    recommendations = []
    for product in recommended_products:
        recommendations.append({
            'product_id': product[0],
            'product_name': product[1],
            'img_link': product[2],
            'product_link': product[3]
        })

    # Render the results on an HTML page
    return render_template('recommendations.html', 
                           entered_product_name=entered_product_name, 
                           entered_product_img_link=entered_product_img_link,
                           entered_product_link=entered_product_link,
                           recommendations=recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
