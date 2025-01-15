from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import pandas as pd
import joblib
import os
import requests
from datetime import datetime


final = joblib.load('recommend/models/final_df.joblib')
product_export=joblib.load('recommend/models/products_export.joblib')

# Create your views here.
def abc(request):
    return HttpResponse('hello world')

def validate_token_and_fetch_details(token, url):
    """
    Validate the given token and fetch details from the provided URL.
    """
    token=token
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        # Make a request to the URL with the token
        response = requests.get(url, headers=headers)

        # Check if the response is successful
        if response.status_code == 200:
            return response.json()  # Return the fetched details
        else:
            # Raise an error if the token is invalid or request fails
            raise ValueError(f"Failed to fetch details. HTTP Status: {response.status_code}")
    except Exception as e:
        raise ValueError(f"Error fetching details: {e}")

def model_creation(request):
    current_directory = os.getcwd()

    # Print the current directory
    print("Current Directory:", current_directory)
    
    order_order=pd.read_csv('recommend/datasets/orders_order.csv')
    order_order_df=order_order[['id','user_id','address_id','status']]
    order_order_df=order_order_df.rename(columns={'id':'order_id'})

    order_orderitem=pd.read_csv('recommend/datasets/orders_orderitem.csv')
    order_orderitem_df=order_orderitem[['price','order_id','product_variant_id','created']]

    order_item=pd.merge(order_order_df,order_orderitem_df,on='order_id')

    product_variant=pd.read_csv('recommend/datasets/product_productvariantcombination.csv')
    product_variant_df=product_variant[['product_id','id']]
    product_variant_df=product_variant_df.rename(columns={'id':'product_variant_id'})


    order_item_product=pd.merge(order_item,product_variant_df,on='product_variant_id')
    order_item_product=order_item_product.sort_values(by='created',ascending=False)
    joblib.dump(order_item_product,"recommend/models/orderitem_product.joblib")

    products_export=pd.read_csv('recommend/datasets/products_export .csv')
    joblib.dump(products_export,"recommend/models/products_export.joblib")


    final=pd.merge(order_item_product,products_export,left_on='product_id',right_on='ID')
    joblib.dump(final,'recommend/models/final_df.joblib')

    return HttpResponse('Sucessfully updated')

def recommended(request,user_id=None):

    recommended_products = []
    if user_id!=None:
        #filtering the dataframe for particular user
        user_df=final[final['user_id']==user_id]


        category_counts=user_df['Category'].value_counts()
        ordered_product_ids = user_df['product_id'].unique()

    # Loop through each category based on its occurrence in the recent orders
        for category, count in category_counts.items():
            # Filter for products in the same category
            category_products = final[final['Category'] == category]

            # Exclude already ordered products
            category_products = category_products[~category_products['ID'].isin(ordered_product_ids)]

            # Optionally, you can sort by 'Created At' or 'Popularity' to get better recommendations
            #category_products_sorted = category_products.sort_values(by='Created At', ascending=False)

            # Add the required number of products for this category based on the count
            recommended_category_products = category_products.head(count)

            # Add the recommended products from this category to the list
            recommended_products.append(recommended_category_products)

            # Step 8: Combine all recommended products into a single DataFrame
        if  recommended_products:
            recommended_products_df = pd.concat(recommended_products)

            # Step 9: Ensure we only return exactly 10 recommended products
            recommended_products_df = recommended_products_df.head(10)

            
            if recommended_products_df.shape[0]!=0:
                return JsonResponse({'similar_products':recommended_products_df.to_dict(orient='records')})
            
            else:
                token = request.headers.get('Authorization')  # Extract token from headers

                url="https://moretrek.com/api/auth/user/all/details/"
                    # Validate token and fetch details from the URL
                token = token.split(" ")[1] if "Bearer " in token else token
                print(token)
                fetched_details = validate_token_and_fetch_details(token, url)

                # Extract data from the JSON response
                user_data = fetched_details.get("data", {})
                user_profile = user_data.get("user_profile", {})

                # Extract gender
                gender = user_profile.get("gender", "Unknown")

                # Extract date of birth and calculate age
                date_of_birth = user_profile.get("date_of_birth")
                if date_of_birth:
                    dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
                    today = datetime.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                else:
                    age = "Unknown"

                print(f"Gender: {gender}")
                print(f"Age: {age}")

                if gender=="MALE":
                    #similar=final[final["Category"]]
                    if age<20:
                        product_export1 = product_export[product_export['Category'].isin(["Boy's Fashion", "Men's Fashion","Fragrances"])]
                        print(product_export1)
                    else:
                        product_export1 = product_export[product_export['Category'].isin(["Men's Fashion", "Men's Watches"])]
                else:
                    product_export1=product_export[product_export['Category'].isin(['Fragrances',"Women's Fashion"])]


                return JsonResponse({'similar_products': product_export1.to_dict(orient='records')})

    else:
        return JsonResponse({'similar_products': product_export.to_dict(orient='records')})          
        