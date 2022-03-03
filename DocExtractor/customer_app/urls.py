from django.urls import path
from .views import LoginAPI,CreateUserView
from knox import views as knox_views

urlpatterns=[
    path('login/',LoginAPI.as_view(),name="login"),
    path('logout/', knox_views.LogoutView.as_view(), name='logout'),
    path('register/',CreateUserView.as_view(),name="register")
]