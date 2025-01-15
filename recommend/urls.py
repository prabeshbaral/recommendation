from django.urls import path
from . import views

urlpatterns=[
    path('abc/',views.abc, name='abc'),
    path('create_model/',views.model_creation, name='model_creation'),
    path('get_recommendation/<str:user_id>',views.recommended,name='recommended'),
    path('get_recommendation/', views.recommended, name='recommended'),  # Without user_id

]