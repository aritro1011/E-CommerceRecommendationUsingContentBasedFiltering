import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords') #only to be done first time


try:
    df = pd.read_csv('amazon.csv')
    
except FileNotFoundError:
    print("Error: The file 'amazon.csv' was not found. Please check the file path.")

# PRELIMINARY DATA CLEANING

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

if 'review_content' in df.columns and 'review_title' in df.columns:
    df = df.dropna(subset=['review_content'])
    df = df.dropna(subset=['review_title'])
   
else:
    print("Error: Columns 'review_content' or 'review_title' do not exist in the dataset.")
if 'user_name' in df.columns:
    df['user_name'] = df['user_name'].fillna("Anonymous")
else:
    print("Column 'user_name' not found in the dataset.")
    
if 'about_product' in df.columns:
    df['about_product'] = df['about_product'].fillna("No Summary")
else:
    print("Column 'about_product' not found in the dataset.")


if 'discount_percentage' in df.columns:
  
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '', regex=False)
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')
else:
    print("Column 'discount_percentage' not found in the dataset.")


df = df.drop_duplicates()

stop_words = set(stopwords.words('english'))

def clean_text(text):
  
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
   
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

if 'review_content' in df.columns:
    df['review_content'] = df['review_content'].apply(clean_text)
else:
    print("Column 'review_content' not found in the dataset.")

if 'about_product' in df.columns:
    df['about_product'] = df['about_product'].apply(clean_text)
else:
    print("Column 'about_product' not found in the dataset.")


currency_column = 'discounted_price'  
if currency_column in df.columns:
    df[currency_column] = df[currency_column].str.replace('â', '', regex=False)
    df[currency_column] = df[currency_column].str.replace(r'[^\d.]', '', regex=True)
    df[currency_column] = pd.to_numeric(df[currency_column], errors='coerce')
else:
    print(f"Column '{currency_column}' not found in the dataset.")


currency_column = 'actual_price' 
if currency_column in df.columns:
    df[currency_column] = df[currency_column].str.replace('â', '', regex=False)
    df[currency_column] = df[currency_column].str.replace(r'[^\d.]', '', regex=True)
    df[currency_column] = pd.to_numeric(df[currency_column], errors='coerce')
else:
    print(f"Column '{currency_column}' not found in the dataset.")
df['rating_count'] = df['rating_count'].fillna('0') 


df['about_product'] = df['about_product'].fillna("No description available")


df = df.drop_duplicates()



try:
    df['rating'] = df['rating'].astype(float)
   
except ValueError:
    print("Non-numeric values found in the 'rating' column. Conversion failed.")


try:
    df['rating_count'] = df['rating_count'].str.replace(',', '').astype(int)
  
except AttributeError:
    print("'rating_count' already cleaned.")
except ValueError:
    print("Non-numeric values found in 'rating_count'. Conversion failed.")

# SCALING THE NUMERICAL FEATURES
from sklearn.preprocessing import MinMaxScaler
numerical_features = ['discounted_price', 'actual_price', 'discount_percentage','rating','rating_count']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# Vectorizing the Textual data
# TF-IDF Vectorization for 'review_content', 'about_product', and 'category'


vectorizer_review_content = TfidfVectorizer(max_features=500)
review_content_tfidf = vectorizer_review_content.fit_transform(df['review_content'])
review_content_tfidf_df = pd.DataFrame(review_content_tfidf.toarray(), 
                                         columns=vectorizer_review_content.get_feature_names_out())


df = pd.concat([df, review_content_tfidf_df], axis=1)



df['about_product'] = df['about_product'].fillna("No description available")


df['about_product'] = df['about_product'].apply(clean_text)


vectorizer_about_product = TfidfVectorizer(max_features=500)
about_product_tfidf = vectorizer_about_product.fit_transform(df['about_product'])
about_product_tfidf_df = pd.DataFrame(about_product_tfidf.toarray(), 
                                        columns=vectorizer_about_product.get_feature_names_out())


df = pd.concat([df, about_product_tfidf_df], axis=1)



if 'category' in df.columns:
    df['category'] = df['category'].fillna("No category available")
else:
    print("Column 'category' not found in the dataset.")


vectorizer_category = TfidfVectorizer(max_features=500)
category_tfidf = vectorizer_category.fit_transform(df['category'])
category_tfidf_df = pd.DataFrame(category_tfidf.toarray(), 
                                   columns=vectorizer_category.get_feature_names_out())


df = pd.concat([df, category_tfidf_df], axis=1)
# Save product_id and product_name before dropping
product_info = df[['product_id', 'product_name','img_link','product_link']]
# Drop the non-numerical columns (besides product_id and product_name)
df=df.drop(columns=['product_id', 'product_name', 'user_id', 'user_name', 'review_id', 'review_title','category','about_product','review_content','img_link','product_link'])
df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].mean())
df['actual_price'] = df['actual_price'].fillna(df['actual_price'].mean())
df['discount_percentage'] = df['discount_percentage'].fillna(df['discount_percentage'].mean())
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['rating_count'] = df['rating_count'].fillna(df['rating_count'].mean())
df = df.fillna(df.mean())
# Add 'product_id' and 'product_name' to X_combined to ensure they're part of the feature set
df_with_name = pd.concat([product_info[['product_id', 'product_name','img_link','product_link']], df], axis=1)


# COSINE SIMILIARITY
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(df)



# FUNCTION
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

# USER INPUT
product_id_input = input("Enter the Product ID (string) to get recommendations: ")
product_id_input = str(product_id_input).strip()

if product_id_input:
    recommended_products = recommend_products(product_id=product_id_input, top_n=5)

    if isinstance(recommended_products, str):
        print(recommended_products)  # If the product ID was not found
    else:
        entered_product_name = df_with_name[df_with_name['product_id'] == product_id_input]['product_name'].values[0]
        entered_product_img_link = df_with_name[df_with_name['product_id'] == product_id_input]['img_link'].values[0]
        entered_product_link = df_with_name[df_with_name['product_id'] == product_id_input]['product_link'].values[0]

        # Display the entered product and the recommended products
        print(f"Entered Product: {entered_product_name} (ID: {product_id_input})")
        print(f"Image Link: {entered_product_img_link}")
        print(f"Product Link: {entered_product_link}")
        print("\nRecommended products (ID, Name, Image Link, Product Link):")

        for product in recommended_products:
            print(f"Product ID: {product[0]}, Product Name: {product[1]}, Image Link: {product[2]}, Product Link: {product[3]}")
            print("\n\n")  # Add two line breaks between each product
