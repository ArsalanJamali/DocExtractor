from django.shortcuts import render
from knox.views import LoginView as KnoxLoginView
from rest_framework.authtoken.serializers import AuthTokenSerializer
from rest_framework import permissions
from django.contrib.auth import login
from rest_framework.response import Response
from rest_framework import generics
from . import serializers

class LoginAPI(KnoxLoginView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        serializer = AuthTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        login(request, user)
        custom=super(LoginAPI, self).post(request, format=None)
        return Response(custom.data)

class CreateUserView(generics.CreateAPIView):
    serializer_class=serializers.UserRegisterSerializer
