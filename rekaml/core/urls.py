from django.urls import path
from core import views
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path('api-token-auth/', obtain_auth_token, name='api_token_auth'),
    path('users/', views.UsersView.as_view(), name='users'),
    path('fine-tune-model/', views.FineTuneModelView.as_view(), name='fine_tune_model'),
]