from django.contrib import admin
from .models import Document,DocumentImage,Label
# Register your models here.

admin.site.register(Document)
admin.site.register(DocumentImage)
admin.site.register(Label)

