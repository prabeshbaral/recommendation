from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import joblib
import os
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-saved models and data
final = joblib.load('recommend/models/final_df.joblib')
product_export = joblib.load('recommend/models/products_export.joblib')
tfidf_matrix = joblib.load('recommend/models/tfidf_matrix.joblib')
tfidf_vectorizer = joblib.load('recommend/models/tfidf_vectorizer.joblib')
product_variantflashsale=joblib.load('recommend/models/product_variantflashsale.joblib')

# Create your views here
def abc(request):
    return HttpResponse('Hello, World!')

def check_for_sale(dataframe, product_variantflashsale=product_variantflashsale):
    """
    This function checks if a product in the dataframe has a matching product_id in the 
    product_variantflashsale table. If a match is found, it calculates the final price 
    based on the discount percentage; otherwise, it sets default values.
    
    Parameters:
    - dataframe (pd.DataFrame): The main dataframe containing product details.
    - product_variantflashsale (pd.DataFrame): The table containing flash sale details.
    
    Returns:
    - pd.DataFrame: Updated dataframe with is_flash, discount_percentage, and final_price columns.
    """
    # Merge the two dataframes on the ID and product_id columns
    merged_df = dataframe.merge(product_variantflashsale, 
                                how='left', 
                                left_on='ID', 
                                right_on='product_id')
    
    # Fill NaN values for is_flash and discount_percentage where no match is found
    merged_df['is_flash'] = merged_df['is_flash'].fillna(False)
    merged_df['discount_percentage'] = merged_df['discount_percentage'].fillna(0)
    
    # Calculate final_price based on discount_percentage
    merged_df['final_price'] = merged_df['Base Price'] * (1 - merged_df['discount_percentage'] / 100)
    
    # Replace final_price with base_price where no flash sale exists
    merged_df['final_price'] = merged_df['final_price'].where(merged_df['is_flash'], merged_df['Base Price'])
    
    # Select only the required columns for the result
    result_df = merged_df[['ID',"Name", "Description",'Base Price',"Category","Image URL",'final_price', 'is_flash', 'discount_percentage']].rename(columns={'ID': 'id',"Name":'name',"Description":'description','Base Price':'base_price','Category':'category','Image URL':'image','is_flash':'is_on_flash_sale'})
    result_df
    
    return result_df
"""
 "id": "20a67ba2-3df5-4619-bb19-3bc77d357dc1",
            "name": "Polo T-shirt",
            "description": "mk.sdagbdjkbjkk",
            "base_price": "200.00",
            "category": "Boy's Fashion",
            "image": null,
            "final_price": 200.0,
            "is_on_flash_sale": false,
            "discount_percentage": 0.0
        },
"""


def validate_token_and_fetch_details(token, url):
    """
    Validate the given token and fetch user details from the provided URL.
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ValueError(f"Error validating token: {e}")

def process_user_profile(user_data):
    """
    Extract and process user profile information such as age and gender.
    """
    user_profile = user_data.get("user_profile", {})
    gender = user_profile.get("gender", "Unknown")
    date_of_birth = user_profile.get("date_of_birth")
    
    age = None
    if date_of_birth:
        try:
            dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
            today = datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            age = "Unknown"
    return gender, age

def fetch_product_recommendations(gender, age):
    """
    Generate product recommendations based on user gender and age.
    """
    if gender == "MALE":
        if age < 20:
            categories = ["Boy's Fashion", "Men's Fashion", "Fragrances"]
        else:
            categories = ["Men's Fashion", "Men's Watches"]
    else:
        categories = ["Fragrances", "Women's Fashion"]

    recommended_products = product_export[product_export['Category'].isin(categories)]
    return recommended_products.head(10)

def model_creation(request):
    """
    Preprocess and save updated data for recommendations.
    """
    # Load datasets
    orders = pd.read_csv('recommend/datasets/orders_order.csv')[['id', 'user_id', 'address_id', 'status']].rename(columns={'id': 'order_id'})
    order_items = pd.read_csv('recommend/datasets/orders_orderitem.csv')[['price', 'order_id', 'product_variant_id', 'created']]
    product_variants = pd.read_csv('recommend/datasets/product_productvariantcombination.csv')[['product_id', 'id']].rename(columns={'id': 'product_variant_id'})
    products_export = pd.read_csv('recommend/datasets/products_export.csv')
    flashsale=pd.read_csv('recommend/datasets/product_variantflashsale.csv')
    product_variantflashsale=pd.merge(product_variants,flashsale,left_on='product_variant_id',right_on='variant_id')




    # Merge datasets
    merged_orders = pd.merge(orders, order_items, on='order_id')
    merged_with_variants = pd.merge(merged_orders, product_variants, on='product_variant_id')
    final_df = pd.merge(merged_with_variants, products_export, left_on='product_id', right_on='ID').sort_values(by='created', ascending=False)
    final_df.rename(columns={'product_id':'id'},inplace=True)

    # Save preprocessed data
    joblib.dump(final_df, 'recommend/models/final_df.joblib')
    joblib.dump(products_export,'recommend/models/products_export.joblib')
    joblib.dump(product_variantflashsale,"recommend/models/product_variantflashsale.joblib")


    #creating model of dataframe
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(products_export['Name'])
    print(tfidf_matrix.shape)

    #creating model of tfidf matrix and vectorizer
    joblib.dump(tfidf_matrix, 'recommend/models/tfidf_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'recommend/models/tfidf_vectorizer.joblib')
    return HttpResponse('Successfully updated!')

def get_similar_products(product_name,product_category, product_id,tfidf_matrix=tfidf_matrix, tfidf_vectorizer=tfidf_vectorizer,df=product_export, top_n=5):
    df_filtered = df[df['Category']==product_category]
    # Get the indices of the filtered rows
    filtered_indices = df_filtered.index

    # Extract the rows from the original TF-IDF matrix corresponding to these indices
    tfidf_matrix_filtered = tfidf_matrix[filtered_indices]

    # Transform the product name into the TF-IDF vector
    product_tfidf = tfidf_vectorizer.transform([product_name])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(product_tfidf, tfidf_matrix_filtered)
    sim_scores = list(enumerate(cosine_sim.flatten()))
    
    # Sort the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    product_indices = [i[0] for i in sim_scores[:top_n+1]]
    
    # Get the top similar products from the filtered DataFrame
    result = df_filtered.iloc[product_indices]
    print(result)


    if ((result['ID']==product_id).any()):

        result=result[result['ID']!=product_id]
    else:
        result=result[:top_n]
    # Return the result without the 'product_id' column
    return result

def recommended(request):


    """
    Generate personalized recommendations for a user based on past purchases or profile.
    """
    token = request.headers.get('Authorization')
    if not token:
        # Default fallback for unauthenticated users
        default_recommendations = product_export.head(10)
        final_prod2=check_for_sale(default_recommendations)
        return JsonResponse({'similar_products': final_prod2.to_dict(orient='records')})
        #return JsonResponse({'error': 'Authorization token missing'}, status=400)

    token = token.split(" ")[1] if "Bearer " in token else token
    url = "https://moretrek.com/api/auth/user/all/details/"
    try:
        user_data = validate_token_and_fetch_details(token, url).get("data", {})
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

    user_id = user_data.get("id")
    if user_id:
        user_df = final[final['user_id'] == user_id]
   
        if not user_df.empty:
            print(user_df.columns)
            # Personalized recommendations based on order history
            category_counts = user_df['Category'].value_counts()
            ordered_product_ids = user_df['id'].unique()
            print(ordered_product_ids,'orderproduct_id')
            print(product_export["ID"].unique)

            recommended_products = []
            for category, count in category_counts.items():
                print(category)
                print(count)
                category_products = product_export[(product_export['Category'] == category) & (~product_export['ID'].isin(ordered_product_ids))]
                recommended_products.append(category_products)

            if recommended_products:
                combined_recommendations = pd.concat(recommended_products)
                combined_recommendations.shape
                combined_recommendations=combined_recommendations
                if combined_recommendations.shape[0]>10:
                    final_prod1=check_for_sale(combined_recommendations).sample(10)
                else:
                    final_prod1=check_for_sale(combined_recommendations)
                return JsonResponse({'similar_products': final_prod1.to_dict(orient='records')})

        # Fallback recommendations based on profile
        gender, age = process_user_profile(user_data)
        fallback_recommendations = fetch_product_recommendations(gender, age)
        final_prod2=check_for_sale(fallback_recommendations)
        return JsonResponse({'similar_products': final_prod2.to_dict(orient='records')})

    # # Default fallback for unauthenticated users
    # default_recommendations = product_export.head(10)
    # return JsonResponse({'similar_products': default_recommendations.to_dict(orient='records')})

def similar_item(request, product_id):
    """
    Fetch similar items based on category of the given product.
    """
    category = product_export.loc[product_export["ID"] == product_id, 'Category'].values
    name=product_export.loc[product_export["ID"] == product_id, 'Name'].values
    similar_products=get_similar_products(name[0],category[0],product_id=product_id)
    final_prod=check_for_sale(similar_products)

    return JsonResponse({'similar_products':final_prod.to_dict(orient='records')})
