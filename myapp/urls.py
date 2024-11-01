from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('features/', views.features, name='features'),
    path('run_utils/', views.run_utils_view, name='run_utils'),  # Add this line
    path('predict-invoice/', views.predict_invoice, name='predict_invoice'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)