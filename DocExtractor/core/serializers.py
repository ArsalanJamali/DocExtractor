from rest_framework import serializers
from .models import Document, DocumentImage

class DocumentSerializer(serializers.ModelSerializer):
    type=serializers.CharField(source='get_type_display')
    image_url=serializers.SerializerMethodField('get_image_url')

    class Meta:
        model=Document
        fields='__all__'
    
    def get_image_url(self,obj):
        doc_images_list=obj.documentimage_set.all()
        if len(doc_images_list)>0:
            doc_image=doc_images_list[0]
            url=doc_image.document_image.url
            return url
        return ''

class DocumentUpdateSerializer(serializers.ModelSerializer):

    class Meta:
        model=Document
        fields=('id','model_id','description')

        extra_kwargs={
            'model_id':{
            'read_only':True,
            },
            'id':{
                'read_only':True,
            }
    }

class DocumentImageSerializer(serializers.ModelSerializer):
    filename=serializers.SerializerMethodField('get_image_filename')


    class Meta:
        model=DocumentImage
        fields='__all__'
    
    def get_image_filename(self,obj):
        return obj.document_image.name.split('/')[1]

        
