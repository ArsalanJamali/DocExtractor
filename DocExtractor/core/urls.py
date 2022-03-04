from django.urls import path
from .views import (create_document_model,get_recent_models,
                    process_documents,save_json,save_csv,UpdateModelDescription,DeleteModelView,
                    DeleteModelImageView,ListModelImages,preview_all_images)

urlpatterns=[
    path('create_document/',create_document_model,name='create'),
    path('get_documents_list/',get_recent_models,name='get_models'),
    path('process_documents/',process_documents,name='process-documents'),
    path('download_json/',save_json,name="download-json"),
    path('download_csv/',save_csv,name='download_csv'),
    path('update_model_description/<int:pk>/',UpdateModelDescription.as_view(),name='update_description'),
    path('delete_model/<int:pk>/',DeleteModelView.as_view(),name='delete_model'),
    path('delete_model_image/<int:pk>/',DeleteModelImageView.as_view(),name='delete_model_image'),
    path('list_model_history/<int:pk>/',ListModelImages.as_view(),name='list_model_images'),
    path('list_all_images/',preview_all_images,name='preview_all_images')

]