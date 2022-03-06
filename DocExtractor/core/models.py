from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import ugettext_lazy as _
import uuid


User=get_user_model()

DOCUMENT_TYPE=(
    (0,'Invoice'),
    (1,'Receipt'),
    (2,'Purchase Order')
)

class Document(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    type=models.IntegerField(choices=DOCUMENT_TYPE,default=0)
    model_id=models.UUIDField(default=uuid.uuid4, editable=False, unique=True,primary_key=False)
    description=models.TextField(blank=True)
    created_at=models.DateField(auto_now_add=True)
    updated_at=models.DateField(auto_now=True)

    def __str__(self):
        return str(self.model_id)

class DocumentImage(models.Model):
    document=models.ForeignKey(Document,on_delete=models.CASCADE)
    document_image=models.ImageField(upload_to='document_images/',null=False)
    upload_date=models.DateField(auto_now_add=True)

    def __str__(self):
        return str(self.document.model_id)

class Label(models.Model):
    image=models.ForeignKey(DocumentImage,on_delete=models.CASCADE)
    key=models.CharField(max_length=100,null=False)
    value=models.TextField(blank=True)

    def __str__(self):
        return str(self.image.pk)