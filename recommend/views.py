from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import joblib
import os
import requests
from datetime import datetime

# Load pre-saved models and data
final = joblib.load('recommend/models/final_df.joblib')
product_export = joblib.load('recommend/models/products_export.joblib')

# Create your views here
def abc(request):
    return HttpResponse('Hello, World!')

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
    products_export = pd.read_csv('recommend/datasets/products_export .csv')

    # Merge datasets
    merged_orders = pd.merge(orders, order_items, on='order_id')
    merged_with_variants = pd.merge(merged_orders, product_variants, on='product_variant_id')
    final_df = pd.merge(merged_with_variants, products_export, left_on='product_id', right_on='ID').sort_values(by='created', ascending=False)

    # Save preprocessed data
    joblib.dump(final_df, 'recommend/models/final_df.joblib')
    joblib.dump(products_export, 'recommend/models/products_export.joblib')
    return HttpResponse('Successfully updated!')

def recommended(request):
    """
    Generate personalized recommendations for a user based on past purchases or profile.
    """
    token = request.headers.get('Authorization')
    if not token:
        return JsonResponse({'error': 'Authorization token missing'}, status=400)

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
            # Personalized recommendations based on order history
            category_counts = user_df['Category'].value_counts()
            ordered_product_ids = user_df['product_id'].unique()

            recommended_products = []
            for category, count in category_counts.items():
                category_products = final[(final['Category'] == category) & (~final['ID'].isin(ordered_product_ids))]
                recommended_products.append(category_products.head(count))

            if recommended_products:
                combined_recommendations = pd.concat(recommended_products).head(10)
                return JsonResponse({'similar_products': combined_recommendations.to_dict(orient='records')})

        # Fallback recommendations based on profile
        gender, age = process_user_profile(user_data)
        print(gender)
        fallback_recommendations = fetch_product_recommendations(gender, age)
        print(fallback_recommendations)
        return JsonResponse({'similar_products': fallback_recommendations.to_dict(orient='records')})

    # Default fallback for unauthenticated users
    default_recommendations = product_export.head(10)
    return JsonResponse({'similar_products': default_recommendations.to_dict(orient='records')})

def similar_item(request, product_id):
    """
    Fetch similar items based on category of the given product.
    """
    category = product_export.loc[product_export["ID"] == product_id, 'Category'].values
    if category:
        similar_products = product_export[product_export['Category'].isin(category)].head(10)
        return JsonResponse({'similar_products': similar_products.to_dict(orient='records')})
    return JsonResponse({'error': 'Product not found'}, status=404)
