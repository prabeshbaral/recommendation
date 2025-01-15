from django.urls import path
from . import views

urlpatterns=[
    path('abc/',views.abc, name='abc'),
    path('create_model/',views.model_creation, name='model_creation'),
    path('get_recommendation/',views.recommended,name='recommended'),
 
    path('similar_item/<str:product_id>',views.similar_item, name='similar_item')

]