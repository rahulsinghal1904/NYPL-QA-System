from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import (handler400, handler403, handler404, handler500)
from DigiSuiteClassifier import views as digisuiteclassifier_views

handler400 = 'DigiSuiteClassifier.views.bad_request'
handler403 = 'DigiSuiteClassifier.views.permission_denied'
handler404 = 'DigiSuiteClassifier.views.page_not_found'
handler500 = 'DigiSuiteClassifier.views.server_error'

urlpatterns=[
    path('', digisuiteclassifier_views.index, name='digisuiteclassifier-home'),
    #path('home', digisuiteclassifier_views.Home, name='home'),
    path('connect-collection/', digisuiteclassifier_views.ConnectCollection, name='connect-collection'),
    path('new-collection/', digisuiteclassifier_views.Home, name='new-collection'),
    #path('new-collection/', digisuiteclassifier_views.NewCollection, name='new-collection'),
    path('update-collection/', digisuiteclassifier_views.UpdateCollection, name='update-collection'),
    path('annotate-collection/', digisuiteclassifier_views.AnnotateCollection, name='annotate-collection'),
    path('export-collection/', digisuiteclassifier_views.ExportCollection, name='export-collection'),
    path('monitor-collection/', digisuiteclassifier_views.CollectionMonitor, name='monitor-collection'),
    path('collection/<int:collection_id>', digisuiteclassifier_views.ViewCollection, name='view-collection'),
    path('custom-collection/', digisuiteclassifier_views.CustomCollection, name='custom-collection'),
    path('view-collection-table', digisuiteclassifier_views.ViewCollectionTable, name='view-collection-table'),
    path('generate_model', digisuiteclassifier_views.GenerateModels, name='generate-models'),
    path('retrain_models', digisuiteclassifier_views.RetrainModels, name='retrain-models'),
    path('chatbot', digisuiteclassifier_views.chatbot_view, name='chatbot'),  # Chat API endpoint
    path('chatbot-nypl', digisuiteclassifier_views.chatbot_view_test, name='chatbot-nypl'),  # Chat API endpoint
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
