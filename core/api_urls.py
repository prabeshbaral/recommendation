from django.urls import path,include


urlpatterns=[
    path('recommend/',include('recommend.urls'))
]