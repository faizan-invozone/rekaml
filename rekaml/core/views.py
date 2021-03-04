from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from rest_framework import status
from utils.MLModel import create_model
from multiprocessing import Process
from pathlib import Path
from rest_framework.parsers import MultiPartParser, FormParser
import os
BASE_DIR = Path(__file__).resolve().parent.parent

class UsersView(APIView):
    
    def post(self, request, format=None):
        res_data = request.data
        try:
            user = User.objects.create(username=res_data['username'], is_active=True, is_staff=True)
            user.set_password(res_data['password'])
            user.save()
            process = Process(
                name='{}_process'.format(user.username), target=create_model, 
                kwargs=({'directory':'{}_{}'.format(user.id, user.username), 'is_new': True})
            )
            process.start()
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_409_CONFLICT)
        return Response({'status': 'created'}, status= status.HTTP_201_CREATED)

class FineTuneModelView(APIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (FormParser, MultiPartParser,)

    def post(self, request, format=None):
        res_file = request.FILES['tuning_file']
        tuning_file_name = res_file.name
        dir = '{}/models/{}_{}/'.format(BASE_DIR, request.user.id, request.user.username)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open('{}/{}'.format(dir, tuning_file_name), 'wb+') as txt_file:
            for line in res_file:
                txt_file.write(line)
        process = Process(
                name='{}_process'.format(request.user.username), target=create_model, 
                kwargs=({
                    'directory':'{}_{}'.format(request.user.id, request.user.username), 
                    'is_new': False,
                    'tuning_file': '{}{}'.format(dir, tuning_file_name)
                })
            )
        process.start()
        return Response({'data': 'received'})
